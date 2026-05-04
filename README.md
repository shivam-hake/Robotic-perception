# Robotic-perception

# CP260-2026 Final Project

## Metric-Semantic Reconstruction

### Authors

Shivam Hake (27141)
Aryan Dahiya (26579)

---

## Overview

This project implements a complete pipeline for metric-semantic reconstruction of a scene using posed RGB images.

The system:

* Reconstructs a 3D scene
* Detects objects using text prompts
* Generates metric-scale oriented bounding boxes (OBB)
* Outputs results in JSON format for evaluation

---

## Key Features

* Depth estimation using Depth Anything
* Open-vocabulary detection using Grounding DINO
* Segmentation via Segment Anything Model (SAM)
* Multi-view 3D reconstruction
* PCA-based oriented bounding boxes
* Automatic scale correction
* Custom entity detection using text prompts

---

## Project Structure

```
.
├── pipeline.py
├── intrinsic.json
├── data/
│   └── scene/
│       ├── frame_*.png
│       └── poses.json
├── outputs/
│   └── answer.json
└── README.md
```

---

## Setup Instructions

### 1. Install Dependencies

```
pip install -r requirements.txt
```

---

### 2. Prepare Dataset

Place images in:

```
data/scene/
```

Ensure the dataset contains:

* frame_XXXXX.png images
* poses.json

---

### 3. Run Pipeline

```
python pipeline.py \
  --data-dir data/scene \
  --intrinsics intrinsic.json \
  --convention auto \
  --max-frames 16 \
  --depth-max 3.0 \
  --check-scale \
  --out-dir outputs \
  --output answer.json \
  --add-entity ethernet_socket "ethernet port" \
  --add-entity power_socket "power connector" \
  --add-entity hdmi_socket_left "left HDMI port below VGA connector" \
  --add-entity usb_socket_top_right "top right USB port above VGA connector"
```

---

## Output

The output is saved as:

```
outputs/answer.json
```

### Format

```
{
  "answers": [
    {
      "label": "ethernet_socket",
      "center": [x, y, z],
      "extent": [dx, dy, dz],
      "R": [[...],[...],[...]]
    }
  ]
}
```

---

## Evaluation

Bounding boxes are evaluated using polygonal Intersection over Union (IoU).
Higher IoU indicates better spatial accuracy.

---

## Important Notes

* Ensure frame IDs match pose IDs
* Always enable scale correction using `--check-scale`
* Use descriptive prompts for better detection accuracy

---

## Experimental Observations

* Scale correction significantly improves IoU
* Multi-view fusion reduces noise
* Prompt engineering improves detection accuracy

---

## Bonus Features

Custom entities can be added using:

```
--add-entity NAME "TEXT QUERY"
```

---

## Challenges Faced

* Pose-image mismatch
* API changes in Grounding DINO
* Incorrect depth scaling
* Sparse dataset handling

---

## References

* Depth Anything V2
* Grounding DINO
* Segment Anything Model (SAM)
* COLMAP
* PCA-based 3D bounding box estimation

---

## Conclusion

This project demonstrates a pipeline that integrates geometric reconstruction with semantic understanding to achieve accurate metric 3D object localization.

---

## Acknowledgment

This project was completed as part of the CP260 (Robotic Perception) course.
