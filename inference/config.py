"""Locked benchmark metrics and class definitions for the wafer defect pipeline."""

from __future__ import annotations

from typing import Dict, List

from wafer_dataset import ID_TO_FAILURE_TYPE, NUM_CLASSES

CLASS_NAMES: List[str] = [ID_TO_FAILURE_TYPE[i] for i in range(NUM_CLASSES)]

# Validation Macro F1-Scores (locked project benchmarks)
BASELINE_METRICS: Dict[str, Dict[str, float | str]] = {
    "resnet34_transfer": {
        "name": "ResNet-34 Transfer Learning",
        "macro_f1": 0.8751,
        "description": "Deep residual pipeline with weighted sampling for class imbalance",
    },
    "hog_random_forest": {
        "name": "HOG + Random Forest",
        "macro_f1": 0.4189,
        "description": "Traditional ML benchmark — failed due to data imbalance blindness",
    },
    "shallow_cnn": {
        "name": "Vanilla Shallow CNN",
        "macro_f1": 0.4313,
        "description": "From-scratch CNN — class collapse and overfitting by epoch 2",
    },
}

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_WEIGHTS_PATH = "baseline_resnet34.pth"
