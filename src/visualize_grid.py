
# src/visualize_grid.py
# Show predictions on 49 test images in a 7x7 grid.
# Great for demo screenshots and README.

import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from config import test_image, best_weights, plots_dir, DEVICE

os.makedirs(plots_dir, exist_ok=True)


def show_grid():
    print('Generating 7x7 prediction grid...')

    if not os.path.exists(best_weights):
        print(f'ERROR: weights not found at {best_weights}')
        print('Run train.py first!')
        return

    model = YOLO(best_weights)

    image_files = [f for f in os.listdir(test_image) if f.lower().endswith('.jpg')]
    step        = max(1, len(image_files) // 49)
    selected    = image_files[::step][:49]

    fig, axes = plt.subplots(7, 7, figsize=(24, 24))
    fig.suptitle('Test Set Predictions — 7×7 Grid', fontsize=20)

    for ax, img_name in zip(axes.flatten(), selected):
        img_path = os.path.join(test_image, img_name)
        img = cv2.imread(img_path)

        if img is not None:
            img_resized = cv2.resize(img, (640, 640))
            results     = model.predict(source=img_resized, imgsz=640, conf=0.3, verbose=False)
            annotated   = results[0].plot(line_width=1)
            annotated   = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            ax.imshow(annotated)
        else:
            ax.text(0.5, 0.5, 'Load failed', ha='center', va='center')

        ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(plots_dir, 'grid_predictions.png')
    plt.savefig(save_path, dpi=80, bbox_inches='tight')
    plt.show()
    print(f'Saved → {save_path}')


if __name__ == '__main__':
    show_grid()
