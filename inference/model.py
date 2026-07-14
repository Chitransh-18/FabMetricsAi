"""Model construction, checkpoint loading, and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

from inference.config import CLASS_NAMES
from wafer_dataset import NUM_CLASSES


def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """Recreate the ResNet-34 backbone with a 9-class classification head."""
    weights = ResNet34_Weights.IMAGENET1K_V1
    model = resnet34(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def resolve_device() -> torch.device:
    """Select CUDA when available, otherwise fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(
    weights_path: str | Path,
    device: torch.device | None = None,
) -> Tuple[nn.Module, torch.device, Dict]:
    """
    Load a training checkpoint and return the model, device, and metadata.

    Safely unpacks the ``model_state_dict`` sub-key from the checkpoint dict.
    """
    device = device or resolve_device()
    path = Path(weights_path)

    if not path.is_file():
        raise FileNotFoundError(f"Model weights not found: {path.resolve()}")

    model = build_model(num_classes=NUM_CLASSES).to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    metadata: Dict = {}
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        metadata = {
            "best_val_f1": checkpoint.get("best_val_f1"),
            "epoch": checkpoint.get("epoch"),
        }
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, device, metadata


@torch.no_grad()
def predict(
    model: nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
) -> Dict:
    """
    Run a single forward pass and return class name, confidence, and full distribution.
    """
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0]
    pred_idx = int(probs.argmax().item())
    confidence = float(probs[pred_idx].item())

    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "predicted_index": pred_idx,
        "confidence": confidence,
        "probabilities": {
            name: float(probs[i].item()) for i, name in enumerate(CLASS_NAMES)
        },
    }
