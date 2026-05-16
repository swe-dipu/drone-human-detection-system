
# src/dataset.py
# Filters VisDrone labels → keeps only pedestrian, people, car
# Remaps class IDs to 0, 1, 2
# Saves cleaned labels to data/processed/

import os
import yaml
from config import (
    BASE, KEEP, REMAP, FINAL_NAMES,
    train_label, valid_label, test_label,
    proc_train_label, proc_valid_label, proc_test_label,
    train_image, valid_image, test_image,
    yaml_path
)


def preprocess_labels(src_label_dir, dst_label_dir, split_name):
    """
    Read every .txt label file from src_label_dir.
    Keep only classes in KEEP, remap their IDs, save to dst_label_dir.
    """
    os.makedirs(dst_label_dir, exist_ok=True)

    files = [f for f in os.listdir(src_label_dir) if f.endswith('.txt')]
    kept_total    = 0
    removed_total = 0

    for file in files:
        src_path = os.path.join(src_label_dir, file)
        dst_path = os.path.join(dst_label_dir, file)

        kept_lines = []

        with open(src_path) as f:
            for line in f:
                values = line.strip().split()
                if len(values) != 5:
                    continue

                cls = int(values[0])

                # Skip classes we don't want
                if cls not in KEEP:
                    removed_total += 1
                    continue

                # Remap class ID (e.g. original 3 → new 2)
                new_cls = REMAP[cls]

                # Coordinates stay exactly the same
                cx, cy, w, h = values[1], values[2], values[3], values[4]
                kept_lines.append(f'{new_cls} {cx} {cy} {w} {h}')
                kept_total += 1

        # Write cleaned file (even if empty — YOLO needs file to exist)
        with open(dst_path, 'w') as f:
            f.write('\n'.join(kept_lines))

    print(f'  [{split_name}]  kept={kept_total}  removed={removed_total}  files={len(files)}')


def create_dataset_yaml():
    """Create data/dataset.yaml that YOLOv8 reads before training."""
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)

    dataset = {
        'path' : BASE,
        'train': train_image,
        'val'  : valid_image,
        'test' : test_image,
        'nc'   : 3,
        'names': FINAL_NAMES
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(dataset, f, default_flow_style=False, sort_keys=False)

    print(f'\n  Saved: {yaml_path}')
    with open(yaml_path) as f:
        print(f.read())


if __name__ == '__main__':
    print('=' * 50)
    print('  VisDrone Label Preprocessing')
    print('=' * 50)

    preprocess_labels(train_label, proc_train_label, 'train')
    preprocess_labels(valid_label, proc_valid_label, 'val')
    preprocess_labels(test_label,  proc_test_label,  'test')

    create_dataset_yaml()
