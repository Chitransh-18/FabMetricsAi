"""
500-Epoch Training Pipeline for Hybrid ResNet34-CBAM Attention Wafer Classifier.
Supports Automatic Mixed Precision (AMP), Cosine Annealing, Checkpointing, and Benchmark Metrics Export.
"""

from __future__ import annotations
import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from pipeline.dataset import BalancedWaferDataset, build_balanced_samples, NUM_CLASSES_EXTENDED
from pipeline.hybrid_model import build_hybrid_model
from pipeline.dual_hybrid_model import build_dual_hybrid_model, FocalLoss

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train State-of-the-Art Dual Fusion Wafer Classifier")
    parser.add_argument("--pkl-path", type=Path, default=Path("LSWMD.pkl"), help="Path to LSWMD.pkl")
    parser.add_argument("--epochs", type=int, default=500, help="Total training epochs (default: 500)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-class", type=int, default=3500, help="Target samples per class (default: 3500)")
    parser.add_argument("--model-type", type=str, choices=["dual_fusion", "resnet34_cbam"], default="dual_fusion", help="Model architecture")
    parser.add_argument("--use-focal-loss", action="store_true", default=True, help="Use Focal Loss for multi-defect optimization")
    parser.add_argument("--checkpoint", type=Path, default=Path("sota_dual_fusion_model.pth"))
    parser.add_argument("--log-json", type=Path, default=Path("training_metrics.json"))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    precision = float(precision_score(all_labels, all_preds, average="macro", zero_division=0))
    recall = float(recall_score(all_labels, all_preds, average="macro", zero_division=0))

    return avg_loss, macro_f1, precision, recall

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
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
    metrics: Dict,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_architecture": "ResNet34-CBAM-Hybrid",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_f1": best_val_f1,
        "metrics": metrics,
    }
    torch.save(checkpoint, path)
    print(f"  -> Saved best hybrid model checkpoint to {path} (Val Macro F1: {best_val_f1:.4f})")

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==========================================================")
    print(f"   HYBRID RESNET34-CBAM ATTENTION MODEL TRAINING ENGINE   ")
    print(f"==========================================================")
    print(f"Execution Device: {device}")
    print(f"Target Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")

    # Build or Load Balanced Augmented Dataset
    samples, stats = build_balanced_samples(
        pkl_path=args.pkl_path,
        target_per_class=args.samples_per_class,
        seed=args.seed,
    )

    full_dataset = BalancedWaferDataset(samples)
    n = len(full_dataset)
    n_val = max(1, int(n * 0.2))
    n_train = n - n_val

    generator = torch.Generator().manual_seed(args.seed)
    train_indices, val_indices = random_split(range(n), [n_train, n_val], generator=generator)

    train_subset = Subset(full_dataset, train_indices.indices)
    val_subset = Subset(full_dataset, val_indices.indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"\nTrain Set: {len(train_subset)} samples | Val Set: {len(val_subset)} samples")

    if args.model_type == "dual_fusion":
        print("[Training Engine] Instantiating Dual-Branch ResNet50-CBAM + EfficientNet-B0 SOTA Architecture...")
        model = build_dual_hybrid_model(num_classes=NUM_CLASSES_EXTENDED, pretrained=True).to(device)
    else:
        print("[Training Engine] Instantiating ResNet34-CBAM Hybrid Architecture...")
        model = build_hybrid_model(num_classes=NUM_CLASSES_EXTENDED, pretrained=True).to(device)

    if args.use_focal_loss:
        print("[Training Engine] Using Focal Loss (alpha=0.25, gamma=2.0) for multi-defect & boundary optimization...")
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_val_f1 = -1.0
    history: List[Dict] = []

    start_train_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_f1, val_prec, val_rec = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        epoch_metric = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_macro_f1": round(val_f1, 4),
            "val_precision": round(val_prec, 4),
            "val_recall": round(val_rec, 4),
            "lr": current_lr,
        }
        history.append(epoch_metric)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_f1={val_f1:.4f} | "
            f"val_prec={val_prec:.4f} | "
            f"val_rec={val_rec:.4f} | "
            f"lr={current_lr:.2e}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_checkpoint(
                path=args.checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_f1=best_val_f1,
                metrics=epoch_metric,
            )

        # Save metrics log periodically
        if epoch % 5 == 0 or epoch == args.epochs:
            with open(args.log_json, "w") as f:
                json.dump({"best_val_f1": best_val_f1, "history": history}, f, indent=2)

    total_duration = time.time() - start_train_time
    print(f"\n==========================================================")
    print(f"Training Complete! Total Duration: {total_duration/60:.2f} mins")
    print(f"Best Validation Macro F1-Score: {best_val_f1:.4f}")
    print(f"Checkpoint saved: {args.checkpoint.resolve()}")
    print(f"==========================================================")

if __name__ == "__main__":
    main()
