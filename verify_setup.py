"""
verify_setup.py
Run this first to confirm your environment is correctly configured.
Usage: python verify_setup.py
"""

import sys
import os

def check(label, fn):
    """Helper to run a check and print pass/fail clearly."""
    try:
        result = fn()
        print(f"  [PASS] {label}: {result}")
        return True
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        return False

print("\n" + "="*55)
print("  Drone Detection Project — Environment Verification")
print("="*55)

print("\n[1] Python & System")
check("Python version", lambda: sys.version.split()[0])
check("Working directory", lambda: os.getcwd())

print("\n[2] Core Libraries")
check("PyTorch", lambda: __import__('torch').__version__)
check("CUDA available", lambda: str(__import__('torch').cuda.is_available()) + " (False is OK for CPU)")
check("OpenCV", lambda: __import__('cv2').__version__)
check("NumPy", lambda: __import__('numpy').__version__)
check("Pandas", lambda: __import__('pandas').__version__)

print("\n[3] ML Libraries")
check("Ultralytics (YOLO)", lambda: __import__('ultralytics').__version__)
check("SAHI", lambda: __import__('sahi').__version__)
check("Albumentations", lambda: __import__('albumentations').__version__)

print("\n[4] Visualization")
check("Matplotlib", lambda: __import__('matplotlib').__version__)
check("Seaborn", lambda: __import__('seaborn').__version__)
check("Pillow", lambda: __import__('PIL').__version__)

print("\n[5] Project Folder Structure")
required_dirs = [
    "data/raw", "data/processed", "data/splits", "data/samples",
    "models/pretrained", "models/finetuned", "models/configs",
    "src", "notebooks",
    "outputs/predictions", "outputs/videos", "outputs/metrics", "outputs/plots",
    "evaluation"
]
all_dirs_ok = True
for d in required_dirs:
    if os.path.isdir(d):
        print(f"  [PASS] {d}/")
    else:
        print(f"  [MISS] {d}/  ← run mkdir command again")
        all_dirs_ok = False

print("\n[6] Quick YOLOv8 Smoke Test")
check(
    "YOLOv8n loads correctly",
    lambda: str(__import__('ultralytics').YOLO('yolov8n.pt'))[:40] + "..."
)

print("\n" + "="*55)
print("  Setup complete! Ready to move to Module 2.")
print("="*55 + "\n")