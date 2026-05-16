# src/config.py
# All paths and settings in ONE place.
# Every other file imports from here — change path once, works everywhere.

import os

# Project Root
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Raw Data Paths (original VisDrone — never modified) ───────
train_image = os.path.join(BASE, 'data/raw/VisDrone2019-DET-train/images')
train_label = os.path.join(BASE, 'data/raw/VisDrone2019-DET-train/labels')

valid_image = os.path.join(BASE, 'data/raw/VisDrone2019-DET-val/images')
valid_label = os.path.join(BASE, 'data/raw/VisDrone2019-DET-val/labels')

test_image  = os.path.join(BASE, 'data/raw/VisDrone2019-DET-test-dev/images')
test_label  = os.path.join(BASE, 'data/raw/VisDrone2019-DET-test-dev/labels')

# Processed Label Paths (filtered + remapped by dataset.py) ─
proc_train_label = os.path.join(BASE, 'data/processed/train/labels')
proc_valid_label = os.path.join(BASE, 'data/processed/val/labels')
proc_test_label  = os.path.join(BASE, 'data/processed/test/labels')

# Output Paths
plots_dir      = os.path.join(BASE, 'outputs/plots')
predictions_dir= os.path.join(BASE, 'outputs/predictions')
metrics_dir    = os.path.join(BASE, 'outputs/metrics')

# Model Paths 
yaml_path      = os.path.join(BASE, 'data/dataset.yaml')
best_weights   = os.path.join(BASE, 'runs/detect/models/finetuned/visdrone_yolov8s/weights/best.pt')

# Class Definitions
VISDRONE_CLASSES = {
    0: 'pedestrian', 1: 'people',   2: 'bicycle',
    3: 'car',        4: 'van',      5: 'truck',
    6: 'tricycle',   7: 'awning-tricycle',
    8: 'bus',        9: 'motor'
}

# Classes we keep (original IDs)
KEEP = {0, 1, 3}

# Remap original → new clean YOLO IDs  (must start from 0, no gaps)
REMAP = {0: 0, 1: 1, 3: 2}

# Final class names after filtering
FINAL_NAMES = ['pedestrian', 'people', 'car']

# Human class IDs in new numbering (used for counting)
HUMAN_IDS = [0, 1]

# Box colors per class — BGR for OpenCV
COLORS_BGR = {
    0: (0,   220, 0),    # pedestrian → green
    1: (0,   140, 255),  # people     → orange
    2: (255, 80,  0),    # car        → blue
}

# Training Settings
EPOCHS     = 30
BATCH_SIZE = 4
IMG_SIZE   = 640
DEVICE     = 'cpu'
