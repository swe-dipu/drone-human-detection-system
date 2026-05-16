
# src/detect.py
# Run inference on test images + count humans per image.
# Saves annotated images to your custom local path.

import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from config import (
    test_image, best_weights,
    FINAL_NAMES, HUMAN_IDS,
    DEVICE
)

# ── Custom save path ──────────────────────────────────────────
SAVE_DIR = 'runs/detect/models/finetuned/visdrone_yolov8s'


os.makedirs(SAVE_DIR, exist_ok=True)


def run_inference():
    print('=' * 50)
    print('  Inference + Human Counting')
    print('=' * 50)
    print(f'  Saving to: {SAVE_DIR}')

    if not os.path.exists(best_weights):
        print(f'ERROR: weights not found at {best_weights}')
        print('Run train.py first!')
        return

    model = YOLO(best_weights)

    # Get first 15 test images
    all_images  = sorted(os.listdir(test_image))[:15]
    image_paths = [os.path.join(test_image, img) for img in all_images]

    for img_path in image_paths:
        # Run inference
        results = model.predict(
            source  = img_path,
            imgsz   = 640,
            conf    = 0.25,
            device  = DEVICE,
            verbose = False
        )

        # Count detections by class
        detected_classes = results[0].boxes.cls.tolist()
        human_count = sum(1 for c in detected_classes if int(c) in HUMAN_IDS)
        car_count   = sum(1 for c in detected_classes if int(c) == 2)

        # Draw boxes using YOLO's built-in plot()
        annotated = results[0].plot(line_width=1)
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        # Add count text on the image
        cv2.putText(annotated, f'Humans: {human_count}  Cars: {car_count}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 50, 50), 2)

        # Save to custom local path
        save_path = os.path.join(SAVE_DIR, os.path.basename(img_path))
        cv2.imwrite(save_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        # Show in notebook/window
        plt.figure(figsize=(16, 8))
        plt.imshow(annotated)
        plt.axis('off')
        plt.title(f'{os.path.basename(img_path)}  |  Humans: {human_count}  Cars: {car_count}')
        plt.tight_layout()
        plt.show()

        print(f'{os.path.basename(img_path):45s}  humans={human_count}  cars={car_count}')

    print(f'\nAll annotated images saved → {SAVE_DIR}')


if __name__ == '__main__':
    run_inference()