"""PyTorch Dataset and DataLoader utilities for the WM-811K wafer map dataset."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import functional as TF

TARGET_SIZE = (224, 224)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FailureType(IntEnum):
    CENTER = 0
    DONUT = 1
    EDGE_LOC = 2
    EDGE_RING = 3
    LOC = 4
    RANDOM = 5
    SCRATCH = 6
    NEAR_FULL = 7
    NONE = 8


FAILURE_TYPE_TO_ID: Dict[str, int] = {
    "Center": FailureType.CENTER,
    "Donut": FailureType.DONUT,
    "Edge-Loc": FailureType.EDGE_LOC,
    "Edge-Ring": FailureType.EDGE_RING,
    "Loc": FailureType.LOC,
    "Random": FailureType.RANDOM,
    "Scratch": FailureType.SCRATCH,
    "Near-full": FailureType.NEAR_FULL,
    "none": FailureType.NONE,
}

ID_TO_FAILURE_TYPE: Dict[int, str] = {v.value: k for k, v in FAILURE_TYPE_TO_ID.items()}

NUM_CLASSES = len(FAILURE_TYPE_TO_ID)


def extract_failure_label(failure_type) -> Optional[str]:
    """Parse nested failureType field, e.g. [['Center']] -> 'Center'."""
    if failure_type is None or (isinstance(failure_type, float) and np.isnan(failure_type)):
        return None
    if isinstance(failure_type, (list, np.ndarray)):
        if len(failure_type) == 0:
            return None
        inner = failure_type[0]
        if isinstance(inner, (list, np.ndarray)):
            return str(inner[0]) if len(inner) > 0 else None
        return str(inner)
    return str(failure_type)


def failure_label_to_id(label: str) -> int:
    try:
        return FAILURE_TYPE_TO_ID[label]
    except KeyError as exc:
        raise ValueError(f"Unknown failure type: {label!r}") from exc


def preprocess_wafer_map(
    wafer_map: np.ndarray,
    merge_die_into_background: bool = False,
) -> np.ndarray:
    """Convert wafer map to float32 HxW in [0, 1]."""
    img = np.asarray(wafer_map, dtype=np.float32)
    if merge_die_into_background:
        img = img.copy()
        img[img == 1] = 0
    img /= 2.0
    return img


@dataclass(frozen=True)
class WaferSample:
    index: int
    wafer_map: np.ndarray
    label_id: int
    label_name: str


def build_wafer_samples(
    pkl_path: Union[str, Path],
    include_none: bool = True,
) -> List[WaferSample]:
    """
    Build the filtered sample index from LSWMD.pkl.

    Mirrors notebook filtering:
    - include_none=True  -> failureNum in [0, 8]  (df_withlabel)
    - include_none=False -> failureNum in [0, 7]  (df_withpattern)
    """
    df = pd.read_pickle(pkl_path)
    max_label = FailureType.NONE if include_none else FailureType.NEAR_FULL

    samples: List[WaferSample] = []
    for idx, row in df.iterrows():
        label_name = extract_failure_label(row["failureType"])
        if label_name is None:
            continue

        try:
            label_id = failure_label_to_id(label_name)
        except ValueError:
            continue

        if label_id < FailureType.CENTER or label_id > max_label:
            continue

        wafer_map = row["waferMap"]
        if wafer_map is None or not hasattr(wafer_map, "shape"):
            continue
        if np.asarray(wafer_map).ndim != 2:
            continue

        samples.append(
            WaferSample(
                index=int(idx),
                wafer_map=np.asarray(wafer_map),
                label_id=label_id,
                label_name=label_name,
            )
        )

    return samples


class WaferTrainTransform:
    """Training-time transforms: flips, 3-channel replication, normalization."""

    def __init__(
        self,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
    ) -> None:
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.mean = mean
        self.std = std

    def __call__(self, image: torch.Tensor, label: int) -> Tuple[torch.Tensor, int]:
        if torch.rand(1).item() < self.hflip_prob:
            image = TF.hflip(image)
        if torch.rand(1).item() < self.vflip_prob:
            image = TF.vflip(image)

        image = image.repeat(3, 1, 1)
        image = TF.normalize(image, mean=list(self.mean), std=list(self.std))
        return image, label


class WaferEvalTransform:
    """Evaluation-time transforms: 3-channel replication and normalization."""

    def __init__(
        self,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
    ) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, image: torch.Tensor, label: int) -> Tuple[torch.Tensor, int]:
        image = image.repeat(3, 1, 1)
        image = TF.normalize(image, mean=list(self.mean), std=list(self.std))
        return image, label


class WaferMapDataset(Dataset):
    """
    PyTorch Dataset for WM-811K wafer maps.

    Loads waferMap matrices, resizes to (224, 224), maps failure types to
    integers 0-8, and applies optional transforms.
    """

    def __init__(
        self,
        pkl_path: Union[str, Path],
        transform: Optional[Callable[[torch.Tensor, int], Tuple[torch.Tensor, int]]] = None,
        target_size: Tuple[int, int] = TARGET_SIZE,
        merge_die_into_background: bool = False,
        include_none: bool = True,
        samples: Optional[List[WaferSample]] = None,
    ) -> None:
        self.pkl_path = Path(pkl_path)
        self.transform = transform
        self.target_size = target_size
        self.merge_die_into_background = merge_die_into_background
        self.include_none = include_none

        if samples is not None:
            self._samples = samples
        else:
            self._samples = build_wafer_samples(
                self.pkl_path,
                include_none=self.include_none,
            )

        if len(self._samples) == 0:
            raise ValueError("No valid samples found after filtering.")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        sample = self._samples[index]

        img = preprocess_wafer_map(
            sample.wafer_map,
            merge_die_into_background=self.merge_die_into_background,
        )

        tensor = torch.from_numpy(img).unsqueeze(0)
        tensor = TF.resize(
            tensor,
            size=list(self.target_size),
            interpolation=transforms.InterpolationMode.NEAREST,
        )

        label = sample.label_id
        if self.transform is not None:
            tensor, label = self.transform(tensor, label)

        return tensor, label

    @property
    def num_classes(self) -> int:
        return NUM_CLASSES if self.include_none else NUM_CLASSES - 1

    @property
    def class_names(self) -> List[str]:
        ids = range(self.num_classes)
        return [ID_TO_FAILURE_TYPE[i] for i in ids]

    def label_distribution(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for sample in self._samples:
            counts[sample.label_name] = counts.get(sample.label_name, 0) + 1
        return counts

    def get_labels(self) -> List[int]:
        return [sample.label_id for sample in self._samples]


@dataclass
class DataLoaderConfig:
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = True
    shuffle_train: bool = True
    val_fraction: float = 0.2
    seed: int = 42
    use_weighted_sampler: bool = False


def create_wafer_dataloaders(
    pkl_path: Union[str, Path],
    config: Optional[DataLoaderConfig] = None,
    merge_die_into_background: bool = False,
) -> Tuple[DataLoader, DataLoader, WaferMapDataset]:
    """
    Create train/validation DataLoaders for 9-class wafer map classification.

    Returns
    -------
    train_loader, val_loader, full_dataset
    """
    config = config or DataLoaderConfig()
    pkl_path = Path(pkl_path)

    samples = build_wafer_samples(pkl_path, include_none=True)
    full_dataset = WaferMapDataset(
        pkl_path=pkl_path,
        include_none=True,
        merge_die_into_background=merge_die_into_background,
        samples=samples,
    )

    n = len(samples)
    n_val = max(1, int(n * config.val_fraction))
    n_train = n - n_val

    generator = torch.Generator().manual_seed(config.seed)
    train_indices, val_indices = torch.utils.data.random_split(
        range(n),
        [n_train, n_val],
        generator=generator,
    )

    train_dataset = WaferMapDataset(
        pkl_path=pkl_path,
        transform=WaferTrainTransform(),
        merge_die_into_background=merge_die_into_background,
        include_none=True,
        samples=samples,
    )
    val_dataset = WaferMapDataset(
        pkl_path=pkl_path,
        transform=WaferEvalTransform(),
        merge_die_into_background=merge_die_into_background,
        include_none=True,
        samples=samples,
    )

    train_subset = Subset(train_dataset, train_indices.indices)
    val_subset = Subset(val_dataset, val_indices.indices)

    sampler = None
    shuffle = config.shuffle_train
    if config.use_weighted_sampler:
        train_labels = [samples[i].label_id for i in train_indices.indices]
        class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = [class_weights[label] for label in train_labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return train_loader, val_loader, full_dataset
