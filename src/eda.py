
# src/eda.py
# Exploratory Data Analysis — run BEFORE training
# Produces 3 plots saved to outputs/plots/

import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter

from config import (
    train_image, train_label,
    valid_image, valid_label,
    test_image,  test_label,
    proc_train_label,
    VISDRONE_CLASSES, FINAL_NAMES, COLORS_BGR, KEEP,
    plots_dir
)

os.makedirs(plots_dir, exist_ok=True)


# 1. Count classes across all splits

def count_classes(label_dir):
    counts = Counter()
    for file in os.listdir(label_dir):
        if file.endswith('.txt'):
            with open(os.path.join(label_dir, file)) as f:
                for line in f:
                    values = line.strip().split()
                    if len(values) == 5:
                        counts[int(values[0])] += 1
    return counts


def show_class_counts():
    train_counts = count_classes(train_label)
    valid_counts = count_classes(valid_label)
    test_counts  = count_classes(test_label)

    print('#' * 60)
    print('Train:', {VISDRONE_CLASSES[k]: v for k, v in sorted(train_counts.items())})
    print('#' * 60)
    print('Valid:', {VISDRONE_CLASSES[k]: v for k, v in sorted(valid_counts.items())})
    print('#' * 60)
    print('Test: ', {VISDRONE_CLASSES[k]: v for k, v in sorted(test_counts.items())})
    print('#' * 60)

    # Build dataframe for seaborn plot
    rows = []
    for cls, count in train_counts.items():
        rows += [(VISDRONE_CLASSES[cls], 'train')] * count
    for cls, count in valid_counts.items():
        rows += [(VISDRONE_CLASSES[cls], 'valid')] * count
    for cls, count in test_counts.items():
        rows += [(VISDRONE_CLASSES[cls], 'test')] * count

    df = pd.DataFrame(rows, columns=['class', 'split'])

    plt.figure(figsize=(15, 6))
    sns.countplot(data=df, x='class', hue='split')
    plt.xticks(rotation=45)
    plt.title('Class Distribution — Train / Validation / Test', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'class_distribution.png'), dpi=100)
    plt.show()
    print(f'  Saved → {plots_dir}/class_distribution.png')



# 2. Show raw sample images with all class boxes

def show_raw_samples():
    image_files = sorted(os.listdir(train_image))[:25]
    class_names_list = list(VISDRONE_CLASSES.values())

    plt.figure(figsize=(20, 20))

    for idx, img_file in enumerate(image_files, 1):
        img_path   = os.path.join(train_image, img_file)
        label_path = os.path.join(train_label, img_file.replace('.jpg', '.txt'))

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    values = line.strip().split()
                    if len(values) == 5:
                        cls, cx, cy, bw, bh = map(float, values)
                        cls = int(cls)
                        x1 = int((cx - bw/2) * w)
                        y1 = int((cy - bh/2) * h)
                        x2 = int((cx + bw/2) * w)
                        y2 = int((cy + bh/2) * h)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(img, class_names_list[cls], (x1, max(15, y1-5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        plt.subplot(5, 5, idx)
        plt.imshow(img)
        plt.axis('off')

    plt.suptitle('VisDrone Raw Samples — All 10 Classes', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'raw_samples.png'), dpi=100, bbox_inches='tight')
    plt.show()
    print(f'  Saved → {plots_dir}/raw_samples.png')



# 3. Check image resolution

def check_image_size():
    sample_name = sorted(os.listdir(train_image))[0]
    sample_path = os.path.join(train_image, sample_name)
    image = cv2.imread(sample_path)
    height, width, channels = image.shape

    print(f'\nImage file : {sample_name}')
    print(f'Height     : {height} px')
    print(f'Width      : {width} px')
    print(f'Channels   : {channels}')
    print(f'When YOLO resizes to 640 → scale = {640/width:.3f}')
    print(f'A 20px person becomes ~{20 * 640/width:.1f}px — very tiny!')



# 4. Show processed samples (after dataset.py runs)

def show_processed_samples():
    if not os.path.exists(proc_train_label):
        print('Run dataset.py first to generate processed labels!')
        return

    image_files = sorted(os.listdir(train_image))[:25]

    plt.figure(figsize=(20, 20))

    for idx, img_file in enumerate(image_files, 1):
        img_path   = os.path.join(train_image, img_file)
        label_path = os.path.join(proc_train_label, img_file.replace('.jpg', '.txt'))

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    values = line.strip().split()
                    if len(values) == 5:
                        cls, cx, cy, bw, bh = map(float, values)
                        cls = int(cls)
                        x1 = int((cx - bw/2) * w)
                        y1 = int((cy - bh/2) * h)
                        x2 = int((cx + bw/2) * w)
                        y2 = int((cy + bh/2) * h)
                        # Use our custom colors per class
                        b, g, r = COLORS_BGR[cls]
                        color_rgb = (r, g, b)
                        cv2.rectangle(img, (x1, y1), (x2, y2), color_rgb, 2)
                        cv2.putText(img, FINAL_NAMES[cls], (x1, max(15, y1-5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_rgb, 1)

        plt.subplot(5, 5, idx)
        plt.imshow(img)
        plt.axis('off')

    legend = [
        mpatches.Patch(color=(0, 220/255, 0),         label='pedestrian'),
        mpatches.Patch(color=(0, 140/255, 255/255),   label='people'),
        mpatches.Patch(color=(255/255, 80/255, 0),    label='car'),
    ]
    plt.figlegend(handles=legend, loc='lower center', ncol=3, fontsize=13)
    plt.suptitle('Processed Labels — pedestrian / people / car only', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'processed_samples.png'), dpi=100, bbox_inches='tight')
    plt.show()
    print(f'  Saved → {plots_dir}/processed_samples.png')



# Run all EDA steps

if __name__ == '__main__':
    print('=' * 50)
    print('  Dataset EDA')
    print('=' * 50)

    print('\n[1] Class counts...')
    show_class_counts()

    print('\n[2] Raw sample images...')
    show_raw_samples()

    print('\n[3] Image resolution check...')
    check_image_size()

    print('\n[4] Processed sample images...')
    show_processed_samples()

    print('\nDone! Check outputs/plots/')
