
# src/train.py
# Fine-tune YOLOv8s on VisDrone dataset


from ultralytics import YOLO
from config import yaml_path, EPOCHS, BATCH_SIZE, IMG_SIZE, DEVICE


def train():
    print('=' * 50)
    print('  YOLOv8s Training — VisDrone')
    print('=' * 50)
    print(f'  Config : {yaml_path}')
    print(f'  Epochs : {EPOCHS}')
    print(f'  Batch  : {BATCH_SIZE}')
    print(f'  Device : {DEVICE}')

    # Load YOLOv8s with pretrained COCO weights
    # Starting from pretrained weights is much faster than scratch
    model = YOLO('yolov8s.pt')

    # Start training
    results = model.train(
        data     = yaml_path,
        epochs   = EPOCHS,
        imgsz    = IMG_SIZE,
        batch    = BATCH_SIZE,
        workers  = 2,
        device   = DEVICE,
        project  = 'models/finetuned',
        name     = 'visdrone_yolov8s',
        exist_ok = True,
        plots    = True,
        verbose  = True
    )

    print('\nTraining complete!')
    print(f'Best weights → {results.save_dir}/weights/best.pt')


if __name__ == '__main__':
    train()
