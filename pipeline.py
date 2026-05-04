#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       CP260-2026  Final Project — Metric-Semantic Reconstruction             ║
║              Shivam Hake(27141)  |  Aryan Dahiya(26579)                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Pipeline
────────
  Step 0 │ Dataset acquire  (GDrive download  OR  local zip  OR  existing dir)
  Step 1 │ Load images · poses · intrinsics
  Step 2 │ Metric depth estimation per frame          (Depth-Anything-V2)
  Step 3 │ Fused coloured scene point-cloud           (Open3D TSDF / voxel)
  Step 4 │ Open-vocabulary 2-D detection per frame    (Grounding-DINO)
  Step 5 │ Precise instance segmentation              (SAM box-prompted)
  Step 6 │ Multi-frame 3-D point gathering  (mask → depth → unproject)
  Step 7 │ Oriented Bounding Box fitting              (Open3D PCA OBB)
  Step 8 │ Validation · IoU self-check · JSON export  →  answer.json

Installation (one-time)
───────────────────────
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  pip install transformers>=4.40 accelerate
  pip install open3d opencv-python Pillow numpy scipy tqdm gdown shapely

Quick-start
───────────
  # Full run — download dataset automatically
  python src/pipeline.py

  # Already have Data.zip locally
  python src/pipeline.py --local-zip /path/to/Data.zip

  # Dataset already extracted
  python src/pipeline.py --no-download --data-dir data/scene

  # Add an extra entity on evaluation day
  python src/pipeline.py --no-download \\
      --add-entity usb_port "USB type-A port socket"

  # Validate your answer.json without re-running inference
  python src/pipeline.py --validate-only --output answer.json

  # Force-rebuild entity point clouds (keeps scene cloud cache)
  python src/pipeline.py --no-download --rebuild-pts
"""

# ─────────────────────────────────────────────────────────────────────────────
#  Standard library
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import json
import logging
import os
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
#  Third-party (guarded imports)
# ─────────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

GDRIVE_FILE_ID = "1U8kTzhToFkHihi6Qw0UTSO2M9_JLo3i"

# ── Baked-in intrinsics from intrinsic.json  (fallback if file not provided) ─
DEFAULT_K = dict(
    fx=1477.00974684544,
    fy=1480.4424455584467,
    cx=1298.2501500778505,
    cy=686.8201623541711,
    W=2560,
    H=1440,
)

# ── HuggingFace model IDs ─────────────────────────────────────────────────────
#   Depth-Anything-V2-Metric-Indoor gives absolute depth in metres for indoor
#   scenes; Small variant runs on CPU if no GPU is available.
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
GDINO_MODEL = "IDEA-Research/grounding-dino-tiny"
SAM_MODEL   = "facebook/sam-vit-base"

# ── Entity text prompts  (period-separated synonym lists for Grounding-DINO) ─
#   Multiple synonyms improve recall on ambiguous objects.
ENTITY_PROMPTS: Dict[str, str] = {
    "vga_socket":      "VGA port . blue trapezoid connector . D-sub 15 pin port",
    "ethernet_socket": "ethernet port . RJ45 jack . LAN socket . network port",
    "power_socket":    "power socket . electrical outlet . AC power connector",
}

# ── Reference OBB  (from sample_answers.json — used for self-validation) ─────
VGA_REFERENCE = {
    "center":   [0.2704921202927293, 0.2261220732082181, 0.8349008829378597],
    "extent":   [0.03537766175069747, 0.011822199241650923, 0.0061316691090621735],
    "rotation": [
        [-0.004004375172752437,  0.9672545151126772, -0.25377680739897346],
        [ 0.01584254528462312,   0.25380835519540434, 0.9671247761234889],
        [ 0.9998664804554559,   -0.00014774012094266, -0.016340117333610394],
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 0 — Dataset acquisition
# ─────────────────────────────────────────────────────────────────────────────

def acquire_dataset(args: argparse.Namespace) -> Path:
    """
    Obtain the dataset by one of three routes (checked in priority order):
      1. --local-zip  : user supplies a local .zip file → extract it
      2. --no-download: dataset already extracted at --data-dir
      3. default      : download from Google Drive then extract

    Returns the path to the extracted scene directory.
    """
    root = Path(args.data_dir)

    # ── Route 1: local zip provided ───────────────────────────────────────────
    if getattr(args, "local_zip", None):
        zip_path = Path(args.local_zip)
        if not zip_path.exists():
            log.error(f"Local zip not found: {zip_path}"); sys.exit(1)
        extract_to = root / "scene"
        if not extract_to.exists():
            log.info(f"Extracting {zip_path} → {extract_to} …")
            extract_to.mkdir(parents=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_to)
        else:
            log.info(f"Already extracted: {extract_to}")
        return extract_to

    # ── Route 2: no download, use existing dir ────────────────────────────────
    if not args.download:
        if not root.exists():
            log.error(f"Data directory not found: {root}"); sys.exit(1)
        return root

    # ── Route 3: Google Drive download ───────────────────────────────────────
    try:
        import gdown
    except ImportError:
        log.error("gdown not installed. Run: pip install gdown"); sys.exit(1)

    root.mkdir(parents=True, exist_ok=True)
    zip_path   = root / "dataset.zip"
    extract_to = root / "scene"

    if not zip_path.exists():
        log.info("Downloading dataset from Google Drive …")
        gdown.download(id=GDRIVE_FILE_ID, output=str(zip_path), quiet=False)
    else:
        log.info(f"Zip already present: {zip_path}")

    if not extract_to.exists():
        log.info("Extracting …")
        extract_to.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)

    return extract_to


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_intrinsics(path: Optional[str]) -> dict:
    """
    Parse camera intrinsics from a JSON file using the camera_matrix format
    produced by OpenCV calibration.  Falls back to baked-in values.
    """
    if path and Path(path).exists():
        raw = json.loads(Path(path).read_text())
        K   = np.asarray(raw["camera_matrix"])
        return dict(
            fx=float(K[0, 0]), fy=float(K[1, 1]),
            cx=float(K[0, 2]), cy=float(K[1, 2]),
            W=int(raw["image_width"]), H=int(raw["image_height"]),
        )
    log.warning("intrinsic.json not found — using baked-in defaults.")
    return DEFAULT_K


# ── Pose parsing ──────────────────────────────────────────────────────────────
# poses.json from NeRF / Colmap / custom SLAM tools can store poses in several
# formats.  We try each in turn and raise if nothing matches.

def _parse_pose_entry(val) -> np.ndarray:
    """
    Convert a single raw pose entry into a 4×4 float64 camera-to-world matrix.

    Supported formats (tried in order):
      A. 4×4 nested list  (standard c2w matrix)
      B. 3×4 nested list  (c2w without bottom row)
      C. flat list of 16  (row-major 4×4)
      D. flat list of 12  (row-major 3×4)
      E. dict with keys 'rotation' (3×3) and 'position' (3,)
      F. dict with keys 'R' and 't'  (could be w2c: we invert)
      G. dict with keys 'transform_matrix' (NeRF blender format)
    """
    if isinstance(val, dict):
        # NeRF blender: {"transform_matrix": [[...],...]}}
        if "transform_matrix" in val:
            return _mat_to_c2w(np.asarray(val["transform_matrix"], dtype=np.float64))
        # {rotation, position}
        if "rotation" in val and "position" in val:
            R   = np.asarray(val["rotation"],  dtype=np.float64).reshape(3, 3)
            t   = np.asarray(val["position"],  dtype=np.float64).reshape(3, 1)
            m34 = np.hstack([R, t])
            return _mat_to_c2w(m34)
        # {R, t}  — likely w2c; invert
        if "R" in val and "t" in val:
            R   = np.asarray(val["R"], dtype=np.float64).reshape(3, 3)
            t   = np.asarray(val["t"], dtype=np.float64).reshape(3, 1)
            # Assume this is world-to-camera; invert to get c2w
            c2w = np.eye(4)
            c2w[:3, :3] = R.T
            c2w[:3,  3] = -(R.T @ t).ravel()
            return c2w
        raise ValueError(f"Unrecognised pose dict keys: {list(val.keys())}")

    arr = np.asarray(val, dtype=np.float64)
    return _mat_to_c2w(arr)


def _mat_to_c2w(m: np.ndarray) -> np.ndarray:
    """Pad a 3×4 or flat-16/12 array to a proper 4×4 c2w matrix."""
    m = m.reshape(-1) if m.ndim == 1 else m
    if m.size == 16:
        return m.reshape(4, 4)
    if m.size == 12:
        m = m.reshape(3, 4)
        return np.vstack([m, [0., 0., 0., 1.]])
    if m.shape == (3, 4):
        return np.vstack([m, [0., 0., 0., 1.]])
    if m.shape == (4, 4):
        return m.copy()
    raise ValueError(f"Cannot convert shape {m.shape} to 4×4 matrix")


def _normalise_fid(k) -> str:
    """
    Normalise a pose key to a zero-padded 3-digit string so it matches
    the frame_000xyz.png filename convention.
    Handles: int 42, str '42', str '042', str 'frame_042'.
    """
    s = str(k)
    # Strip a leading 'frame_' prefix if present
    if s.startswith("frame_"):
        s = s[len("frame_"):]
    # Keep only digits and zero-pad to 3
    digits = "".join(c for c in s if c.isdigit())
    if not digits:
        return s          # fall back to raw key if no digits found
    return digits.zfill(3)


def load_poses(path: str) -> Dict[str, np.ndarray]:
    """
    Load all camera poses from poses.json.
    Returns dict[frame_id (3-digit str) → 4×4 float64 c2w matrix].
    """
    raw = json.loads(Path(path).read_text())

    # The JSON could be a dict {fid: pose} or a list [{fid, pose}, ...]
    if isinstance(raw, list):
        items = [(str(i), entry) for i, entry in enumerate(raw)]
    else:
        items = list(raw.items())

    poses: Dict[str, np.ndarray] = {}
    skipped = 0
    for k, v in items:
        fid = _normalise_fid(k)
        try:
            poses[fid] = _parse_pose_entry(v)
        except Exception as exc:
            log.debug(f"Skipping pose key={k!r}: {exc}")
            skipped += 1

    log.info(f"Loaded {len(poses)} poses  (skipped {skipped}) from {path}")
    return poses


def _discover_images(scene_dir: Path) -> List[Path]:
    """Recursively search for frame_*.png/jpg up to 2 directory levels."""
    for depth in range(3):
        glob_base = "/".join(["*"] * depth) + ("/" if depth else "")
        for ext in ("png", "jpg", "jpeg"):
            hits = sorted(scene_dir.glob(f"{glob_base}frame_*.{ext}"))
            if hits:
                return hits
    # Fallback: any images
    for ext in ("png", "jpg", "jpeg"):
        hits = sorted(scene_dir.rglob(f"*.{ext}"))
        if hits:
            return hits
    return []


def load_images(scene_dir: Path) -> Tuple[Dict[str, np.ndarray], Path]:
    """
    Load all frame images from scene_dir.
    Returns (dict[frame_id → RGB uint8 array], image directory path).
    """
    paths = _discover_images(scene_dir)
    if not paths:
        raise FileNotFoundError(f"No images found under {scene_dir}")

    img_dir = paths[0].parent
    imgs: Dict[str, np.ndarray] = {}
    for p in paths:
        fid = _normalise_fid(p.stem)
        bgr = cv2.imread(str(p))
        if bgr is not None:
            imgs[fid] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    log.info(f"Loaded {len(imgs)} images from {img_dir}")
    return imgs, img_dir


# ─────────────────────────────────────────────────────────────────────────────
#  Coordinate-convention helpers
# ─────────────────────────────────────────────────────────────────────────────

# NeRF / Blender uses OpenGL convention  (Y-up, Z toward camera / backward).
# Open3D and depth unprojection use OpenCV convention  (Y-down, Z forward).
# If poses are in OpenGL convention we must flip Y and Z axes.

OPENGL_TO_OPENCV = np.array(
    [[1,  0,  0, 0],
     [0, -1,  0, 0],
     [0,  0, -1, 0],
     [0,  0,  0, 1]],
    dtype=np.float64,
)


def convert_poses_to_opencv(poses: Dict[str, np.ndarray], convention: str) -> Dict[str, np.ndarray]:
    """
    Convert c2w matrices from `convention` to OpenCV.

    Args:
        convention: 'opencv'  — no-op
                    'opengl'  — flip Y and Z (NeRF/Blender format)
                    'auto'    — heuristic: inspect whether average camera
                                look-direction has Z positive or negative
    """
    if convention == "opencv":
        return poses

    if convention == "auto":
        # Average look direction in world coords is the 3rd column of R in c2w.
        # In OpenCV (Z-forward) the camera looks along +Z in camera space;
        # transformed to world, the Z of the look-direction should be positive
        # when cameras face inward to a desk.  If median Z < 0, assume OpenGL.
        look_z = [c2w[2, 2] for c2w in poses.values()]
        median_z = float(np.median(look_z))
        log.info(f"Convention auto-detect: median look-dir Z = {median_z:.3f}")
        if median_z < 0:
            convention = "opengl"
            log.info("  → Treating as OpenGL convention; flipping Y & Z.")
        else:
            convention = "opencv"
            log.info("  → Treating as OpenCV convention; no flip needed.")

    if convention == "opengl":
        return {fid: c2w @ OPENGL_TO_OPENCV for fid, c2w in poses.items()}

    return poses


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — Metric depth estimation
# ─────────────────────────────────────────────────────────────────────────────

class DepthEstimator:
    """
    Depth-Anything-V2-Metric-Indoor via HuggingFace depth-estimation pipeline.

    The Metric-Indoor variant is calibrated to return absolute depth in metres
    for indoor scenes (desktops, rooms) without any scale ambiguity.
    The Small variant runs on CPU within reasonable time.
    """

    def __init__(self, model_id: str = DEPTH_MODEL):
        from transformers import pipeline as hfpipe
        device = 0 if torch.cuda.is_available() else -1
        log.info(
            f"Loading depth model [{model_id}] "
            f"on {'CUDA' if device == 0 else 'CPU'} …"
        )
        self._pipe = hfpipe(
            task="depth-estimation",
            model=model_id,
            device=device,
        )
        log.info("Depth estimator ready.")

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """
        Args:  rgb : (H, W, 3) uint8 RGB image
        Returns: depth : (H, W) float32 metric depth in metres
        """
        result = self._pipe(Image.fromarray(rgb))
        # 'predicted_depth' is the raw metric tensor (not the 0-1 normalised image)
        depth = result["predicted_depth"].squeeze().float().numpy()
        H, W  = rgb.shape[:2]
        if depth.shape != (H, W):
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
        return depth.astype(np.float32)


def run_depth_estimation(
    images:    Dict[str, np.ndarray],
    cache_dir: Path,
    estimator: DepthEstimator,
    scale:     float = 1.0,   # global depth scale correction (applied before saving)
) -> Dict[str, np.ndarray]:
    """
    Estimate depth for all frames.  Caches results as .npy files so
    subsequent runs skip inference.  Scale correction is baked into the cache.
    """
    cache_dir.mkdir(exist_ok=True)
    depths: Dict[str, np.ndarray] = {}

    for fid, img in tqdm(images.items(), desc="Depth estimation"):
        cache = cache_dir / f"{fid}.npy"
        if cache.exists():
            depths[fid] = np.load(str(cache))
        else:
            d = estimator(img) * scale
            np.save(str(cache), d)
            depths[fid] = d

    log.info(f"Depth maps ready for {len(depths)} frames.")
    return depths


def apply_depth_scale(
    depths:    Dict[str, np.ndarray],
    scale:     float,
    cache_dir: Path,
) -> Dict[str, np.ndarray]:
    """
    Apply a depth scale factor to all depth maps and overwrite the cache.
    This ensures the correction persists across runs.
    """
    if abs(scale - 1.0) < 1e-6:
        return depths
    log.info(f"Applying depth scale {scale:.5f} and overwriting cache …")
    scaled: Dict[str, np.ndarray] = {}
    for fid, d in depths.items():
        sd = (d * scale).astype(np.float32)
        np.save(str(cache_dir / f"{fid}.npy"), sd)
        scaled[fid] = sd
    return scaled


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — Scene point cloud
# ─────────────────────────────────────────────────────────────────────────────

def _unproject_frame(
    depth: np.ndarray,
    K:     dict,
    c2w:   np.ndarray,
    rgb:   Optional[np.ndarray] = None,
    d_min: float = 0.05,
    d_max: float = 3.0,
) -> o3d.geometry.PointCloud:
    """
    Back-project a depth map into 3-D world frame using the pinhole model.

    Camera-frame:
        X_c = (u - cx) / fx * d
        Y_c = (v - cy) / fy * d
        Z_c = d

    World-frame:   P_w = c2w @ [X_c  Y_c  Z_c  1]^T
    """
    H, W  = depth.shape
    fx, fy, cx, cy = K["fx"], K["fy"], K["cx"], K["cy"]

    u, v  = np.meshgrid(np.arange(W, dtype=np.float32),
                        np.arange(H, dtype=np.float32))
    valid = (depth > d_min) & (depth < d_max) & np.isfinite(depth)
    d_v   = depth[valid].astype(np.float64)

    X = (u[valid] - cx) / fx * d_v
    Y = (v[valid] - cy) / fy * d_v
    Z = d_v

    pts_h = np.stack([X, Y, Z, np.ones_like(Z)], axis=1)  # (N, 4)
    pts_w = (c2w @ pts_h.T).T[:, :3]                       # (N, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_w)
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb[valid] / 255.0)
    return pcd


def build_scene_cloud(
    images: Dict[str, np.ndarray],
    depths: Dict[str, np.ndarray],
    poses:  Dict[str, np.ndarray],
    K:      dict,
    d_max:  float = 3.0,
    voxel:  float = 0.004,
    stride: int   = 5,
) -> o3d.geometry.PointCloud:
    """
    Fuse depth maps from all frames into one coloured point cloud.
    Voxel downsampling + statistical outlier removal keep it tractable.
    """
    log.info("Building scene point cloud …")
    combined  = o3d.geometry.PointCloud()
    valid_fids = sorted(set(images) & set(depths) & set(poses))

    for fid in tqdm(valid_fids[::stride], desc="Fusing frames"):
        pcd = _unproject_frame(
            depths[fid], K, poses[fid],
            rgb=images[fid], d_max=d_max,
        )
        combined += pcd
        if len(combined.points) > 6_000_000:
            combined = combined.voxel_down_sample(voxel)

    combined = combined.voxel_down_sample(voxel)
    combined, _ = combined.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    log.info(f"Scene cloud: {len(combined.points):,} points")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
#  STEPS 4 & 5 — Open-vocabulary detection + segmentation
# ─────────────────────────────────────────────────────────────────────────────

class Segmenter:
    """
    Two-stage open-vocabulary segmenter:
      Stage 1 — Grounding-DINO   : text → axis-aligned bounding boxes + scores
      Stage 2 — SAM (box-prompt) : box → precise binary instance mask

    Text format:  "synonym1 . synonym2 . synonym3"
    Multiple synonyms increase recall on visually similar / rarely-trained objects.

    If Grounding-DINO returns zero detections with the compound query,
    we automatically retry with each synonym individually (fallback).
    """

    def __init__(self):
        from transformers import (
            AutoProcessor,
            AutoModelForZeroShotObjectDetection,
            SamModel,
            SamProcessor,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        log.info(f"Loading segmentation models on {device.upper()} …")

        log.info(f"  Grounding-DINO  [{GDINO_MODEL}]")
        self.gd_proc  = AutoProcessor.from_pretrained(GDINO_MODEL)
        self.gd_model = (
            AutoModelForZeroShotObjectDetection
            .from_pretrained(GDINO_MODEL)
            .to(device)
        )

        log.info(f"  SAM             [{SAM_MODEL}]")
        self.sam_proc  = SamProcessor.from_pretrained(SAM_MODEL)
        self.sam_model = SamModel.from_pretrained(SAM_MODEL).to(device)

        log.info("Segmenter ready.")

    @torch.no_grad()
    def _gdino(
        self,
        pil:      Image.Image,
        query:    str,
        box_thr:  float = 0.25,
        text_thr: float = 0.25,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run Grounding-DINO; return (boxes_xyxy, scores)."""
        H, W = pil.height, pil.width
        inp  = self.gd_proc(images=pil, text=query,
                            return_tensors="pt").to(self.device)
        out  = self.gd_model(**inp)
        res  = self.gd_proc.post_process_grounded_object_detection(
            out,
            inp.input_ids,
            target_sizes=[(H, W)],
        )[0]
        boxes  = res["boxes"].cpu().float().numpy()
        scores = res["scores"].cpu().float().numpy()
        return boxes, scores

    @torch.no_grad()
    def _sam(
        self,
        pil:   Image.Image,
        boxes: np.ndarray,
    ) -> np.ndarray:
        """Run SAM with box prompts; return (N, H, W) bool mask array."""
        inp = self.sam_proc(pil, input_boxes=[boxes.tolist()],
                            return_tensors="pt").to(self.device)
        out = self.sam_model(**inp)
        raw = self.sam_proc.post_process_masks(
            out.pred_masks.cpu(),
            inp["original_sizes"].cpu(),
            inp["reshaped_input_sizes"].cpu(),
        )[0]  # (N, 3, H, W)
        iou_s = out.iou_scores.cpu().numpy()[0]   # (N, 3)
        best  = iou_s.argmax(axis=1)              # (N,)
        return np.stack([raw[i, best[i]].numpy().astype(bool)
                         for i in range(len(boxes))])

    def detect_and_segment(
        self,
        rgb:      np.ndarray,
        query:    str,
        box_thr:  float = 0.25,
        text_thr: float = 0.25,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect + segment objects matching `query` in `rgb`.

        Fallback: if the compound query detects nothing, retry each
        period-separated synonym individually and merge results.

        Returns:
            boxes  : (N, 4) float32  xyxy pixel coords
            scores : (N,)   float32  Grounding-DINO confidence
            masks  : (N, H, W) bool  SAM instance masks
        """
        pil   = Image.fromarray(rgb)
        boxes, scores = self._gdino(pil, query, box_thr, text_thr)

        # ── Fallback: try each synonym individually ───────────────────────────
        if len(boxes) == 0:
            synonyms = [s.strip() for s in query.split(".") if s.strip()]
            for syn in synonyms:
                b, s = self._gdino(pil, syn, box_thr=0.20, text_thr=0.20)
                if len(b):
                    boxes  = b
                    scores = s
                    log.debug(f"  Fallback query '{syn}' found {len(b)} box(es)")
                    break

        if len(boxes) == 0:
            return np.array([]), np.array([]), np.array([])

        masks = self._sam(pil, boxes)
        return boxes, scores, masks

    def best_detection(
        self,
        rgb:      np.ndarray,
        query:    str,
        min_area: int = 80,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Return the (mask, box) of the single highest-scoring detection, or
        (None, None) if nothing passes the area threshold.
        """
        boxes, scores, masks = self.detect_and_segment(rgb, query)
        if len(masks) == 0:
            return None, None

        areas   = masks.sum(axis=(1, 2))
        valid   = areas >= min_area
        if not valid.any():
            return None, None

        weighted = scores * valid.astype(float)
        idx      = int(weighted.argmax())
        return masks[idx], boxes[idx]


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 6 — Multi-frame 3-D point gathering
# ─────────────────────────────────────────────────────────────────────────────

def _mask_to_world(
    mask:  np.ndarray,
    depth: np.ndarray,
    c2w:   np.ndarray,
    K:     dict,
    d_min: float = 0.05,
    d_max: float = 5.0,
) -> np.ndarray:
    """
    Back-project pixels inside `mask` to 3-D world coordinates.
    Returns (N, 3) float64 world-frame points.
    """
    H, W = depth.shape
    if mask.shape != (H, W):
        mask = cv2.resize(
            mask.astype(np.uint8), (W, H),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    fx, fy, cx, cy = K["fx"], K["fy"], K["cx"], K["cy"]
    valid = mask & (depth > d_min) & (depth < d_max) & np.isfinite(depth)
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return np.empty((0, 3))

    d     = depth[ys, xs].astype(np.float64)
    X     = (xs - cx) / fx * d
    Y     = (ys - cy) / fy * d
    Z     = d
    pts_h = np.stack([X, Y, Z, np.ones_like(Z)], axis=1)
    return (c2w @ pts_h.T).T[:, :3]


def gather_entity_points(
    entity:      str,
    query:       str,
    images:      Dict[str, np.ndarray],
    depths:      Dict[str, np.ndarray],
    poses:       Dict[str, np.ndarray],
    K:           dict,
    segmenter:   Segmenter,
    max_frames:  int   = 10,
    d_max:       float = 3.0,
) -> Tuple[np.ndarray, Dict[str, Tuple]]:
    """
    Collect 3-D world points for `entity` by:
      - Uniformly subsampling up to `max_frames` frames
      - Detecting + segmenting the entity in each frame
      - Back-projecting masked depth pixels to 3-D

    Returns:
        pts          : (N, 3) float64  raw world-frame points
        per_frame_det: dict[fid → (mask, box)]  — kept for viz reuse
    """
    valid_fids = sorted(set(images) & set(depths) & set(poses))
    stride     = max(1, len(valid_fids) // max_frames)
    sampled    = valid_fids[::stride][:max_frames]

    log.info(f"[{entity}] Scanning {len(sampled)} frames …")
    all_pts:    List[np.ndarray] = []
    per_frame:  Dict[str, Tuple] = {}

    for fid in tqdm(sampled, desc=entity, leave=False):
        mask, box = segmenter.best_detection(images[fid], query)
        if mask is None:
            continue
        per_frame[fid] = (mask, box)
        pts = _mask_to_world(mask, depths[fid], poses[fid], K, d_max=d_max)
        if len(pts):
            all_pts.append(pts)

    if not all_pts:
        log.warning(f"[{entity}] No 3-D points found in any frame.")
        return np.empty((0, 3)), per_frame

    out = np.vstack(all_pts)
    log.info(f"[{entity}] Gathered {len(out):,} raw 3-D points "
             f"from {len(per_frame)} frames.")
    return out, per_frame


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 7 — Oriented Bounding Box fitting
# ─────────────────────────────────────────────────────────────────────────────

def fit_obb(pts: np.ndarray) -> Dict:
    """
    Fit a PCA-based Oriented Bounding Box to a set of 3-D points.

    Processing:
      1. Two-pass statistical outlier removal (removes depth noise and
         segmentation bleed-through at object boundaries).
      2. Open3D's get_oriented_bounding_box() (PCA; minimum-volume box).

    Output dict:
        center   : [x, y, z]          world-frame centroid
        extent   : [ex, ey, ez]       full axis lengths (largest first)
        rotation : [[r00,r01,r02],    columns = principal axes (orthonormal)
                    [r10,r11,r12],    R satisfies R@R^T=I, det(R)=+1
                    [r20,r21,r22]]
    """
    if len(pts) < 4:
        raise ValueError(f"OBB fitting needs ≥4 points; got {len(pts)}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.5)

    if len(pcd.points) < 4:
        pcd.points = o3d.utility.Vector3dVector(pts)

    obb = pcd.get_oriented_bounding_box()
    return {
        "center":   np.asarray(obb.center).tolist(),
        "extent":   np.asarray(obb.extent).tolist(),
        "rotation": np.asarray(obb.R).tolist(),
    }


def refine_obb_with_scene(
    obb_init:  Dict,
    scene_pcd: o3d.geometry.PointCloud,
    expand:    float = 1.5,
) -> Dict:
    """
    Crop the full scene cloud to the OBB region (expanded by `expand`) and
    re-fit.  Helps when segmentation-based points are sparse or noisy.
    """
    if len(scene_pcd.points) < 10:
        return obb_init

    ctr    = np.asarray(obb_init["center"])
    R_mat  = np.asarray(obb_init["rotation"])
    ext    = np.asarray(obb_init["extent"])

    crop_box = o3d.geometry.OrientedBoundingBox(
        center=ctr, R=R_mat, extent=ext * expand,
    )
    cropped  = scene_pcd.crop(crop_box)

    if len(cropped.points) < 10:
        log.debug("Scene crop too small — keeping initial OBB.")
        return obb_init

    try:
        return fit_obb(np.asarray(cropped.points))
    except Exception as exc:
        log.debug(f"Scene refinement failed: {exc}")
        return obb_init


# ─────────────────────────────────────────────────────────────────────────────
#  Depth-scale sanity check  (anchored on VGA socket physical size)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_depth_scale(
    depths:     Dict[str, np.ndarray],
    images:     Dict[str, np.ndarray],
    poses:      Dict[str, np.ndarray],
    K:          dict,
    segmenter:  Segmenter,
    n_frames:   int = 5,
) -> float:
    """
    Estimate depth scale from the VGA socket physical width.

    The D-sub 15-pin VGA connector housing is 35.37 mm wide (exact value from
    sample_answers.json extent[0]).  We detect the VGA socket in a few frames,
    back-project its pixels, compute the PCA extent along the primary axis, and
    compare with the physical reference.

    Returns scale factor  (depth_true = depth_estimated * scale).
    Returns 1.0 if estimation is unreliable (outside [0.5, 2.0]).
    """
    VGA_WIDTH_M = 0.03537766175069747   # from sample_answers.json

    query   = ENTITY_PROMPTS["vga_socket"]
    valid   = sorted(set(images) & set(depths) & set(poses))
    sampled = valid[::max(1, len(valid) // n_frames)][:n_frames]

    widths: List[float] = []
    for fid in sampled:
        mask, _ = segmenter.best_detection(images[fid], query)
        if mask is None:
            continue
        pts = _mask_to_world(mask, depths[fid], poses[fid], K, d_max=5.0)
        if len(pts) < 20:
            continue
        centered = pts - pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(centered.T @ centered / len(pts))
        proj     = centered @ Vt[0]
        widths.append(float(proj.max() - proj.min()))

    if not widths:
        log.warning("Depth scale estimation: VGA not detected; returning 1.0")
        return 1.0

    estimated = float(np.median(widths))
    if estimated < 1e-6:
        return 1.0

    scale = VGA_WIDTH_M / estimated
    log.info(
        f"Depth scale — estimated VGA width: {estimated*100:.2f} cm  "
        f"(expected {VGA_WIDTH_M*100:.2f} cm)  →  scale = {scale:.4f}"
    )
    return float(scale) if 0.5 < scale < 2.0 else 1.0


# ─────────────────────────────────────────────────────────────────────────────
#  Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def project_obb_corners(
    obb: Dict,
    K:   dict,
    c2w: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project all 8 OBB corners into image space.

    Returns:
        uv       : (8, 2) float32  pixel coordinates
        in_front : (8,)   bool     True if corner is in front of camera
    """
    ctr = np.asarray(obb["center"])
    ext = np.asarray(obb["extent"]) / 2.0
    R   = np.asarray(obb["rotation"])

    signs = np.array([[s0, s1, s2]
                      for s0 in (1, -1)
                      for s1 in (1, -1)
                      for s2 in (1, -1)], dtype=np.float64)  # (8, 3)
    corners_w = (R @ (signs * ext).T).T + ctr                # (8, 3)

    w2c   = np.linalg.inv(c2w)
    ch    = np.hstack([corners_w, np.ones((8, 1))])
    cc    = (w2c @ ch.T).T[:, :3]                            # (8, 3) camera frame

    in_front = cc[:, 2] > 1e-3
    z_safe   = np.where(cc[:, 2] > 1e-3, cc[:, 2], 1e-3)
    fx, fy, cx, cy = K["fx"], K["fy"], K["cx"], K["cy"]
    u = cc[:, 0] / z_safe * fx + cx
    v = cc[:, 1] / z_safe * fy + cy

    return np.stack([u, v], axis=1).astype(np.float32), in_front


def draw_obb_wireframe(
    obb:    Dict,
    K:      dict,
    c2w:    np.ndarray,
    rgb:    np.ndarray,
    color:  Tuple = (0, 230, 0),
    thick:  int   = 3,
) -> np.ndarray:
    """Overlay a projected 3-D OBB wireframe onto an RGB image."""
    uv, in_front = project_obb_corners(obb, K, c2w)
    EDGES = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),
             (4,5),(4,6),(5,7),(6,7)]
    vis = rgb.copy()
    for i, j in EDGES:
        if in_front[i] and in_front[j]:
            cv2.line(vis, tuple(uv[i].astype(int)), tuple(uv[j].astype(int)),
                     color, thick)
    return vis


def save_detection_mosaic(
    rgb:    np.ndarray,
    mask:   np.ndarray,
    box:    np.ndarray,
    label:  str,
    path:   str,
    scale:  float = 0.35,
) -> None:
    """Save a side-by-side original | detection overlay image."""
    H, W    = rgb.shape[:2]
    overlay = rgb.copy()
    tmp     = overlay.copy()
    tmp[mask] = (255, 80, 80)
    overlay = cv2.addWeighted(overlay, 0.55, tmp, 0.45, 0)
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 3)
    cv2.putText(overlay, label, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
    mosaic = np.concatenate([rgb, overlay], axis=1)
    nH, nW = int(H * scale), int(W * 2 * scale)
    cv2.imwrite(path, cv2.cvtColor(
        cv2.resize(mosaic, (nW, nH)), cv2.COLOR_RGB2BGR))


# ─────────────────────────────────────────────────────────────────────────────
#  Output format, validation, IoU
# ─────────────────────────────────────────────────────────────────────────────

_NULL_OBB = {
    "center":   [0.0, 0.0, 0.0],
    "extent":   [0.01, 0.01, 0.01],
    "rotation": [[1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0]],
}


def build_output(entity_obbs: Dict[str, Optional[Dict]]) -> List[Dict]:
    """Format entity OBBs into the required submission JSON list."""
    return [
        {"entity": name, "obb": obb if obb is not None else _NULL_OBB}
        for name, obb in entity_obbs.items()
    ]


def validate_output(records: List[Dict]) -> bool:
    """
    Strict schema validation for the submission JSON.
    Returns True only if ALL records are valid.

    Checks:
      - Required keys present
      - center and extent have exactly 3 float values
      - rotation is exactly 3×3 (catches the 2-row bug in some editors)
      - All numbers are finite
    """
    ok = True
    for rec in records:
        name = rec.get("entity", "<unknown>")
        if "entity" not in rec:
            log.error("Record missing 'entity' key"); ok = False; continue
        if "obb" not in rec:
            log.error(f"[{name}] missing 'obb' key"); ok = False; continue

        obb = rec["obb"]
        for key, n in [("center", 3), ("extent", 3)]:
            v = obb.get(key, [])
            if len(v) != n:
                log.error(f"[{name}] '{key}' must have exactly {n} values, got {len(v)}")
                ok = False
            elif not all(np.isfinite(x) for x in v):
                log.error(f"[{name}] '{key}' contains non-finite values")
                ok = False

        rot = obb.get("rotation", [])
        if len(rot) != 3:
            log.error(f"[{name}] 'rotation' must have exactly 3 rows, got {len(rot)}")
            ok = False
        else:
            for ri, row in enumerate(rot):
                if len(row) != 3:
                    log.error(f"[{name}] rotation row {ri} must have 3 values, got {len(row)}")
                    ok = False
                elif not all(np.isfinite(x) for x in row):
                    log.error(f"[{name}] rotation row {ri} contains non-finite values")
                    ok = False

        # Check rotation is approximately orthonormal
        if ok:
            R = np.asarray(obb["rotation"])
            if R.shape == (3, 3):
                orth_err = np.linalg.norm(R @ R.T - np.eye(3))
                if orth_err > 0.01:
                    log.warning(f"[{name}] Rotation matrix may not be orthonormal "
                                f"(||R@R^T - I|| = {orth_err:.4f})")

    if ok:
        log.info("JSON validation ✓  — format is correct.")
    else:
        log.error("JSON validation ✗  — fix errors before submission!")
    return ok


def compute_projected_iou(
    obb_a: Dict,
    obb_b: Dict,
    K:     dict,
    c2w:   np.ndarray,
) -> float:
    """
    2-D polygonal IoU of two OBBs projected onto an image plane.
    Requires shapely (pip install shapely).
    Returns NaN if shapely is unavailable or projection is degenerate.
    """
    try:
        from shapely.geometry import Polygon
    except ImportError:
        return float("nan")

    def _poly(obb):
        uv, in_front = project_obb_corners(obb, K, c2w)
        pts = uv[in_front]
        if len(pts) < 3:
            return None
        hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
        if len(hull) < 3:
            return None
        return Polygon(hull)

    pa, pb = _poly(obb_a), _poly(obb_b)
    if pa is None or pb is None or not pa.is_valid or not pb.is_valid:
        return float("nan")
    inter = pa.intersection(pb).area
    union = pa.union(pb).area
    return inter / union if union > 1e-6 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    out_dir    = Path(args.out_dir)
    depth_dir  = out_dir / "depths"
    cloud_path = out_dir / "scene.pcd"
    out_dir.mkdir(exist_ok=True)

    # ── Validate-only mode ────────────────────────────────────────────────────
    if args.validate_only:
        if not Path(args.output).exists():
            log.error(f"File not found: {args.output}"); sys.exit(1)
        records = json.loads(Path(args.output).read_text())
        validate_output(records)
        return

    # ── STEP 0: Dataset acquisition ───────────────────────────────────────────
    scene_root = acquire_dataset(args)

    # Locate poses.json
    pose_files = list(scene_root.rglob("poses.json"))
    if not pose_files:
        log.error(f"poses.json not found under {scene_root}"); sys.exit(1)
    poses_path = pose_files[0]
    scene_dir  = poses_path.parent
    log.info(f"Scene directory : {scene_dir}")

    # ── STEP 1: Load ──────────────────────────────────────────────────────────
    K      = load_intrinsics(args.intrinsics)
    poses  = load_poses(str(poses_path))
    images, _ = load_images(scene_dir)

    # ── Coordinate convention ─────────────────────────────────────────────────
    poses = convert_poses_to_opencv(poses, args.convention)

    common = sorted(set(images) & set(poses))
    log.info(f"Frames with both image and pose: {len(common)}")
    if not common:
        log.error("No common frame IDs — check that poses.json keys match "
                  "frame filenames (both should be 3-digit numbers).")
        sys.exit(1)

    # ── STEP 2: Depth estimation ──────────────────────────────────────────────
    estimator = DepthEstimator(DEPTH_MODEL)
    depths    = run_depth_estimation(images, depth_dir, estimator)

    # ── Depth scale check (BEFORE building scene cloud) ───────────────────────
    segmenter = Segmenter()    # load once; reused throughout

    if args.check_scale:
        scale = estimate_depth_scale(depths, images, poses, K, segmenter)

        # FORCE SCALE APPLICATION
        log.info(f"Forcing depth scale correction: {scale}")
        depths = apply_depth_scale(depths, scale, depth_dir)

    # ── STEP 3: Scene cloud ───────────────────────────────────────────────────
    if cloud_path.exists() and not args.rebuild_cloud:
        log.info("Loading cached scene cloud …")
        scene_pcd = o3d.io.read_point_cloud(str(cloud_path))
    else:
        scene_pcd = build_scene_cloud(
            images, depths, poses, K,
            d_max=args.depth_max,
            stride=args.cloud_stride,
        )
        o3d.io.write_point_cloud(str(cloud_path), scene_pcd)
        log.info(f"Scene cloud saved → {cloud_path}")

    # ── Build entity list (defaults + CLI additions) ──────────────────────────
    entities: Dict[str, str] = dict(ENTITY_PROMPTS)
    for (name, query) in (args.add_entity or []):
        entities[name] = query
        log.info(f"Added entity: {name!r} → {query!r}")

    # ── STEPS 4-7: Detect → Segment → Gather → OBB ───────────────────────────
    entity_obbs:   Dict[str, Optional[Dict]] = {}
    entity_frames: Dict[str, Dict]           = {}   # for viz reuse

    for entity, query in entities.items():
        log.info(f"\n{'━'*64}\n  Entity : {entity}\n  Query  : {query}\n{'━'*64}")

        pts_cache = out_dir / f"pts_{entity}.npy"
        if pts_cache.exists() and not args.rebuild_pts:
            pts       = np.load(str(pts_cache))
            per_frame = {}
            log.info(f"[{entity}] Loaded {len(pts):,} cached points.")
        else:
            pts, per_frame = gather_entity_points(
                entity, query, images, depths, poses, K,
                segmenter,
                max_frames=args.max_frames,
                d_max=args.depth_max,
            )
            np.save(str(pts_cache), pts)

        entity_frames[entity] = per_frame

        if len(pts) < 10:
            log.error(f"[{entity}] Too few points ({len(pts)}) — OBB skipped.")
            entity_obbs[entity] = None
            continue

        try:
            obb = fit_obb(pts)
            obb = refine_obb_with_scene(obb, scene_pcd)
            entity_obbs[entity] = obb
            log.info(f"[{entity}]  center = {[f'{v:.4f}' for v in obb['center']]}")
            log.info(f"[{entity}]  extent = {[f'{v:.5f}' for v in obb['extent']]}")
        except Exception as exc:
            log.error(f"[{entity}] OBB fitting failed: {exc}")
            entity_obbs[entity] = None

        # ── Visualisation (reuse per_frame detections — no extra inference) ───
        if not args.no_viz and entity_obbs.get(entity) and per_frame:
            fid0 = next(iter(per_frame))
            mask, box = per_frame[fid0]

            save_detection_mosaic(
                images[fid0], mask, box, entity,
                str(out_dir / f"det_{entity}_{fid0}.jpg"),
            )
            obb_vis = draw_obb_wireframe(
                entity_obbs[entity], K, poses[fid0], images[fid0],
            )
            cv2.imwrite(
                str(out_dir / f"obb_{entity}_{fid0}.jpg"),
                cv2.cvtColor(obb_vis, cv2.COLOR_RGB2BGR),
            )

    # ── STEP 8: Export ────────────────────────────────────────────────────────
    records = build_output(entity_obbs)

    if not validate_output(records):
        log.error("Aborting — fix validation errors before submission.")
        sys.exit(1)

    out_path = Path(args.output)
    out_path.write_text(json.dumps(records, indent=2))
    log.info(f"\n{'='*64}\n  Answer written → {out_path.resolve()}\n{'='*64}")

    # ── Self-validation: VGA socket IoU against ground-truth reference ────────
    vga_obb = entity_obbs.get("vga_socket")
    if vga_obb is not None and common:
        fid0 = common[0]
        iou  = compute_projected_iou(vga_obb, VGA_REFERENCE, K, poses[fid0])
        if not np.isnan(iou):
            log.info(f"\n  VGA socket self-check IoU (frame {fid0}): {iou:.4f}"
                     f"  {'✓ GOOD' if iou > 0.5 else '✗ LOW — check scale / convention'}")
        else:
            log.info("  (pip install shapely for IoU computation)")

    # ── Summary table ─────────────────────────────────────────────────────────
    log.info(f"\n{'─'*80}")
    log.info(f"  {'Entity':<22}  {'Center (m)':<44}  Extent (m)")
    log.info(f"{'─'*80}")
    for rec in records:
        c = rec["obb"]["center"]
        e = rec["obb"]["extent"]
        log.info(
            f"  {rec['entity']:<22}  "
            f"[{c[0]:+.4f}, {c[1]:+.4f}, {c[2]:+.4f}]  "
            f"[{e[0]:.5f}, {e[1]:.5f}, {e[2]:.5f}]"
        )
    log.info(f"{'─'*80}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline.py",
        description="CP260-2026 Metric-Semantic Reconstruction — OBB Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data acquisition ──────────────────────────────────────────────────────
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--local-zip",   default=None,
                     help="Path to local dataset zip (skip GDrive download)")
    grp.add_argument("--no-download", dest="download", action="store_false",
                     help="Dataset already extracted at --data-dir")
    p.add_argument("--data-dir",    default="data",
                   help="Root directory for download / extracted scene")
    p.add_argument("--intrinsics",  default="intrinsic.json",
                   help="Path to intrinsic.json")

    # ── Coordinate convention ─────────────────────────────────────────────────
    p.add_argument(
        "--convention",
        choices=["auto", "opencv", "opengl"],
        default="auto",
        help=(
            "Camera pose coordinate convention: 'opencv' (Y-down, Z-forward), "
            "'opengl' (Y-up, Z-backward / NeRF Blender), or 'auto' to detect."
        ),
    )

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument("--output",  default="answer.json",
                   help="Path to write the final OBB answer JSON")
    p.add_argument("--out-dir", default="outputs",
                   help="Directory for intermediate outputs (depths, cloud, viz)")

    # ── Entity control ────────────────────────────────────────────────────────
    p.add_argument(
        "--add-entity", nargs=2, metavar=("NAME", "QUERY"),
        action="append",
        help=(
            "Add an entity at runtime. "
            "Example: --add-entity hdmi_port 'HDMI socket port'  "
            "(Repeat for multiple entities.)"
        ),
    )

    # ── Computation parameters ────────────────────────────────────────────────
    p.add_argument("--max-frames",   type=int,   default=10,
                   help="Max frames scanned per entity for 3-D point gathering")
    p.add_argument("--depth-max",    type=float, default=3.0,
                   help="Depth clipping plane in metres")
    p.add_argument("--cloud-stride", type=int,   default=5,
                   help="Frame stride for scene cloud fusion")

    # ── Cache control ─────────────────────────────────────────────────────────
    p.add_argument("--rebuild-cloud", action="store_true",
                   help="Force rebuild scene cloud (ignore cache)")
    p.add_argument("--rebuild-pts",   action="store_true",
                   help="Force re-gather entity point clouds (ignore cache)")

    # ── Optional features ─────────────────────────────────────────────────────
    p.add_argument("--check-scale", action="store_true",
                   help="Estimate + correct depth scale using VGA socket size")
    p.add_argument("--no-viz",      action="store_true",
                   help="Disable visualisation image output")
    p.add_argument("--validate-only", action="store_true",
                   help="Only validate an existing answer.json (no inference)")

    p.set_defaults(download=True)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
