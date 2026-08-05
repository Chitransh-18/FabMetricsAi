"""Modular Machine Learning & Research Pipeline for Wafer Defect Detection."""

import os
from pathlib import Path

# Redirect PyTorch and HuggingFace caches to D: drive to protect C: drive space
D_CACHE_DIR = Path("D:/Web Dev/FabMetrics_AI/cache")
D_CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TORCH_HOME"] = str(D_CACHE_DIR / "torch")
os.environ["HF_HOME"] = str(D_CACHE_DIR / "huggingface")

__version__ = "2.0.0"
