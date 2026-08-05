"""PyTorch Dataset & DataLoader implementation for balanced wafer maps with Multi-Defect support."""

from __future__ import annotations
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.transforms import functional as TF

from wafer_dataset import (
    FAILURE_TYPE_TO_ID,
    ID_TO_FAILURE_TYPE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    TARGET_SIZE,
    extract_failure_label,
    failure_label_to_id,
    preprocess_wafer_map,
)
from pipeline.augment import generate_random_augmentation, create_multi_defect_wafer

EXTENDED_FAILURE_TYPE_TO_ID: Dict[str, int] = {
    **FAILURE_TYPE_TO_ID,
    "Multi-Defect": 9
}

EXTENDED_ID_TO_FAILURE_TYPE: Dict[int, str] = {
    **ID_TO_FAILURE_TYPE,
    9: "Multi-Defect"
}

NUM_CLASSES_EXTENDED = len(EXTENDED_FAILURE_TYPE_TO_ID)

class BalancedWaferDataset(Dataset):
    """
    PyTorch Dataset for 10-class Balanced WM-811K Wafer Maps (including Multi-Defect samples).
    """

    def __init__(
        self,
        samples: List[Tuple[np.ndarray, int, str]],
        transform: Optional[Any] = None,
        target_size: Tuple[int, int] = TARGET_SIZE,
    ) -> None:
        self.samples = samples
        self.transform = transform
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        wafer_map, label_id, _ = self.samples[index]

        img = preprocess_wafer_map(wafer_map)
        tensor = torch.from_numpy(img).unsqueeze(0)
        tensor = TF.resize(
            tensor,
            size=list(self.target_size),
            interpolation=transforms.InterpolationMode.NEAREST,
        )

        tensor = tensor.repeat(3, 1, 1)
        if self.transform is not None:
            tensor = self.transform(tensor)
        else:
            tensor = TF.normalize(tensor, mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD))

        return tensor, label_id

    def label_distribution(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for _, label_id, _ in self.samples:
            name = EXTENDED_ID_TO_FAILURE_TYPE[label_id]
            counts[name] = counts.get(name, 0) + 1
        return counts


def build_balanced_samples(
    pkl_path: Union[str, Path],
    target_per_class: int = 3500,
    include_multi_defect: bool = True,
    seed: int = 42,
) -> Tuple[List[Tuple[np.ndarray, int, str]], Dict[str, Any]]:
    """
    Load LSWMD.pkl, parse labeled failure patterns, apply targeted augmentations,
    and generate multi-defect superimpositions so that every class has `target_per_class` samples.
    """
    random.seed(seed)
    np.random.seed(seed)
    pkl_path = Path(pkl_path)
    d_drive_pkl = Path("D:/Web Dev/FabMetrics_AI/LSWMD.pkl")

    if not pkl_path.exists():
        if d_drive_pkl.exists():
            pkl_path = d_drive_pkl
        else:
            raise FileNotFoundError(f"Dataset pickle file not found at {pkl_path} or {d_drive_pkl}")


    print(f"[Dataset Engine] Loading raw wafer dataset from {pkl_path.name}...")
    df = pd.read_pickle(pkl_path)

    raw_class_samples: Dict[int, List[np.ndarray]] = {i: [] for i in range(8)}  # 8 defect classes + none
    raw_class_samples[8] = []  # 'none'

    for _, row in df.iterrows():
        label_name = extract_failure_label(row["failureType"])
        if label_name is None:
            continue
        try:
            label_id = failure_label_to_id(label_name)
        except ValueError:
            continue

        wafer_map = row["waferMap"]
        if wafer_map is None or not hasattr(wafer_map, "shape"):
            continue
        arr = np.asarray(wafer_map)
        if arr.ndim != 2:
            continue

        raw_class_samples[label_id].append(arr)

    original_counts = {EXTENDED_ID_TO_FAILURE_TYPE[i]: len(raw_class_samples[i]) for i in range(9)}
    print(f"[Dataset Engine] Original Class Distribution: {original_counts}")

    balanced_samples: List[Tuple[np.ndarray, int, str]] = []
    final_counts: Dict[str, int] = {}

    # Single Defect & 'none' Classes (0..8)
    for label_id in range(9):
        name = EXTENDED_ID_TO_FAILURE_TYPE[label_id]
        pool = raw_class_samples[label_id]

        if len(pool) == 0:
            print(f"Warning: Class {name!r} has no samples.")
            continue

        sampled_originals = pool if len(pool) <= target_per_class else random.sample(pool, target_per_class)
        for arr in sampled_originals:
            balanced_samples.append((arr, label_id, "Original"))

        curr_count = len(sampled_originals)
        while curr_count < target_per_class:
            base_arr = random.choice(pool)
            aug_arr, aug_desc = generate_random_augmentation(base_arr)
            balanced_samples.append((aug_arr, label_id, aug_desc))
            curr_count += 1

        final_counts[name] = curr_count

    # Multi-Defect Synthesis Class (ID = 9)
    if include_multi_defect:
        defect_class_ids = [0, 1, 2, 3, 4, 5, 6, 7] # Exclude 'none'
        multi_count = 0
        multi_samples: List[Tuple[np.ndarray, int, str]] = []
        
        while multi_count < target_per_class:
            c1_id, c2_id = random.sample(defect_class_ids, 2)
            w1 = random.choice(raw_class_samples[c1_id])
            w2 = random.choice(raw_class_samples[c2_id])
            
            multi_map = create_multi_defect_wafer(w1, w2)
            c1_name = EXTENDED_ID_TO_FAILURE_TYPE[c1_id]
            c2_name = EXTENDED_ID_TO_FAILURE_TYPE[c2_id]
            desc = f"Superimposed Dual-Defect ({c1_name} + {c2_name})"
            
            multi_samples.append((multi_map, 9, desc))
            multi_count += 1
            
        balanced_samples.extend(multi_samples)
        final_counts["Multi-Defect"] = multi_count

    stats = {
        "original_counts": original_counts,
        "balanced_counts": final_counts,
        "total_samples": len(balanced_samples),
        "target_per_class": target_per_class
    }

    print(f"[Dataset Engine] Equalized Balanced Distribution ({len(final_counts)} Classes): {final_counts}")
    print(f"[Dataset Engine] Total Balanced Samples: {len(balanced_samples)}")

    return balanced_samples, stats
