"""Input preprocessing for wafer map inference."""

from __future__ import annotations

import io
from typing import List, Sequence, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

from wafer_dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    TARGET_SIZE,
    preprocess_wafer_map,
)


def _array_to_tensor(wafer_map: np.ndarray) -> torch.Tensor:
    """Convert a 2-D wafer map array into a normalized 3-channel ResNet tensor."""
    if wafer_map.ndim != 2:
        raise ValueError(f"wafer_map must be 2-D, got shape {wafer_map.shape}")

    img = preprocess_wafer_map(wafer_map)
    tensor = torch.from_numpy(img).unsqueeze(0)
    tensor = TF.resize(
        tensor,
        size=list(TARGET_SIZE),
        interpolation=transforms.InterpolationMode.NEAREST,
    )
    tensor = tensor.repeat(3, 1, 1)
    tensor = TF.normalize(tensor, mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD))
    return tensor


def preprocess_from_array(
    wafer_map: Union[np.ndarray, Sequence[Sequence[float]]],
) -> torch.Tensor:
    """Preprocess a raw wafer map matrix for ResNet inference."""
    arr = np.asarray(wafer_map, dtype=np.float32)
    if arr.size == 0:
        raise ValueError("wafer_map array is empty")
    return _array_to_tensor(arr)


def _decode_image_to_wafer_map(image: Image.Image) -> np.ndarray:
    """
    Decode an uploaded image into a wafer map matrix.

    Grayscale images are used directly. RGB images are converted to luminance.
    Pixel values are mapped to the WM-811K encoding (0, 1, 2) when possible.
    """
    if image.mode not in ("L", "RGB", "RGBA", "P"):
        image = image.convert("L")
    elif image.mode == "RGB":
        image = image.convert("L")
    elif image.mode in ("RGBA", "P"):
        image = image.convert("L")

    arr = np.asarray(image, dtype=np.float32)

    unique = np.unique(arr)
    if set(unique.tolist()).issubset({0.0, 1.0, 2.0}):
        return arr

    if arr.max() <= 2.0:
        return arr

    # Scale 8-bit visualizations back to WM-811K die states.
    scaled = np.zeros_like(arr)
    scaled[arr > 170] = 2.0
    scaled[(arr > 80) & (arr <= 170)] = 1.0
    return scaled


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode uploaded image bytes into a 2-D wafer map matrix."""
    if not data:
        raise ValueError("Uploaded file is empty")

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:
        raise ValueError("Invalid or unsupported image file") from exc

    return _decode_image_to_wafer_map(image)


def preprocess_from_image_bytes(data: bytes) -> torch.Tensor:
    """Preprocess an uploaded image file for ResNet inference."""
    wafer_map = decode_image_bytes(data)
    return _array_to_tensor(wafer_map)


def wafer_map_preview(wafer_map: np.ndarray) -> List[List[int]]:
    """Return a compact integer grid for frontend visualization."""
    preview = np.asarray(wafer_map, dtype=np.int32)
    return preview.tolist()
