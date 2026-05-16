# =============================================================
# evaluation/evaluate.py
# Evaluate trained model on the test set.
# Prints mAP50, mAP50-95, Precision, Recall
#
# Run: python evaluation/evaluate.py
# =============================================================
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ultralytics import YOLO
from config import yaml_path, best_weights, DEVICE


def evaluate():
    print('=' * 50)
    print('  Model Evaluation on Test Set')
    print('=' * 50)

    if not __import__('os').path.exists(best_weights):
        print(f'ERROR: weights not found at {best_weights}')
        print('Run train.py first!')
        return

    model = YOLO(best_weights)

    # Run validation on test split
    metrics = model.val(
        data   = yaml_path,
        split  = 'test',
        imgsz  = 640,
        device = DEVICE
    )

    # Print all results
    print('\n' + '=' * 50)
    print('  Results')
    print('=' * 50)
    for name, value in metrics.results_dict.items():
        print(f'  {name:<35}: {value:.4f}')


if __name__ == '__main__':
    evaluate()
