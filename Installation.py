# @title 1. Install dependencies  {display-mode: "form"}
import subprocess, sys

def run(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-2000:])
    return result.returncode == 0

print("Installing PyTorch ...")
run("pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118")

print("Installing other dependencies ...")
run("pip install -q transformers>=4.40 accelerate open3d opencv-python Pillow numpy scipy tqdm gdown shapely")

print(" All dependencies installed.")
