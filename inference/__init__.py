"""Inference utilities for the wafer defect classifier."""

from inference.config import BASELINE_METRICS, CLASS_NAMES, DEFAULT_WEIGHTS_PATH
from inference.model import build_model, load_model, predict, resolve_device
from inference.preprocess import (
    localize_defects,
    preprocess_from_array,
    preprocess_from_image_bytes,
)

__all__ = [
    "BASELINE_METRICS",
    "CLASS_NAMES",
    "DEFAULT_WEIGHTS_PATH",
    "build_model",
    "load_model",
    "localize_defects",
    "predict",
    "preprocess_from_array",
    "preprocess_from_image_bytes",
    "resolve_device",
]

