"""Train a ResNet34 baseline on WM-811K wafer map failure classification."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import resnet34, ResNet34_Weights

from wafer_dataset import (
    NUM_CLASSES,
    DataLoaderConfig,
    create_wafer_dataloaders,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet34 wafer classifier")
    parser.add_argument(
        "--pkl-path",
        type=Path,
        default=Path("LSWMD.pkl"),
        help="Path to LSWMD.pkl",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("baseline_resnet34.pth"),
        help="Checkpoint path saved when validation Macro-F1 improves",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    weights = ResNet34_Weights.IMAGENET1K_V1
    model = resnet34(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    for images, labels in tqdm(loader, desc="Validating", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_val_f1: float,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_f1": best_val_f1,
    }
    torch.save(checkpoint, path)
    tqdm.write(f"  -> Saved checkpoint to {path}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = DataLoaderConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_fraction=0.2,  # 80% train / 20% validation
        seed=args.seed,
        shuffle_train=True,
        pin_memory=torch.cuda.is_available(),
        use_weighted_sampler=True,
    )

    train_loader, val_loader, dataset = create_wafer_dataloaders(
        pkl_path=args.pkl_path,
        config=config,
    )

    print(f"Dataset size: {len(dataset)} samples")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"Class distribution: {dataset.label_distribution()}")
    print("Using WeightedRandomSampler for class-balanced training batches")

    model = build_model(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_f1 = -1.0

    for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs"):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_macro_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        tqdm.write(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_macro_f1={val_macro_f1:.4f} | "
            f"lr={current_lr:.2e}"
        )

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            save_checkpoint(
                path=args.checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_f1=best_val_f1,
            )

    print(f"\nTraining complete. Best validation Macro-F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()