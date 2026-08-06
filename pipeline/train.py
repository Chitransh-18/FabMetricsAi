"""
State-of-the-Art Training & Evaluation Pipeline for Semiconductor Wafer Defect Classifiers.
Supports Automatic Mixed Precision (AMP), Cosine Annealing, Focal Loss, SWA,
Per-Class Metrics, and Automated Confusion Matrix Artifact Generation.
"""

from __future__ import annotations
import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report

from pipeline.dataset import (
    BalancedWaferDataset,
    build_balanced_samples,
    EXTENDED_ID_TO_FAILURE_TYPE,
    NUM_CLASSES_EXTENDED
)
from pipeline.hybrid_model import build_hybrid_model
from pipeline.dual_hybrid_model import build_dual_hybrid_model, FocalLoss

CLASS_NAMES_10 = [EXTENDED_ID_TO_FAILURE_TYPE[i] for i in range(10)]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train State-of-the-Art Dual Fusion Wafer Classifier")
    parser.add_argument("--pkl-path", type=Path, default=Path("LSWMD.pkl"), help="Path to LSWMD.pkl")
    parser.add_argument("--epochs", type=int, default=2, help="Total training epochs (default: 2 for dry-run)")
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
    parser.add_argument("--cm-png", type=Path, default=Path("confusion_matrix.png"))
    parser.add_argument("--cm-json", type=Path, default=Path("confusion_matrix.json"))
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
) -> Tuple[float, float, float, float, List[int], List[int]]:
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

    return avg_loss, macro_f1, precision, recall, all_labels, all_preds

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any | None,
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


def render_confusion_matrix_image(
    cm: np.ndarray,
    class_names: List[str],
    out_path: Path
) -> None:
    """
    Render a publication-quality colorized confusion matrix heatmap image using OpenCV.
    No matplotlib dependencies required for headless server execution.
    """
    num_classes = len(class_names)
    cell_size = 80
    margin_left = 140
    margin_bottom = 120
    margin_top = 80
    margin_right = 40

    width = margin_left + num_classes * cell_size + margin_right
    height = margin_top + num_classes * cell_size + margin_bottom

    # Dark Slate Background (#0f172a)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (42, 23, 15)

    # Title Header
    cv2.putText(
        img,
        "CONFUSION MATRIX - FABMETRICS AI (10-CLASS WAFER MODEL)",
        (margin_left - 20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (235, 235, 235),
        2,
        cv2.LINE_AA
    )

    cm_norm = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    for i in range(num_classes):
        # Y Axis Labels (True Class)
        label_text = class_names[i]
        cv2.putText(
            img,
            label_text,
            (15, margin_top + i * cell_size + cell_size // 2 + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (203, 213, 225),
            1,
            cv2.LINE_AA
        )

        for j in range(num_classes):
            x1 = margin_left + j * cell_size
            y1 = margin_top + i * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size

            val = cm[i, j]
            norm_val = cm_norm[i, j]

            # Dynamic Heatmap BGR Color
            if i == j:
                # Diagonal (Correct Predictions) -> Vibrant Emerald to Cyan
                b = int(120 + 135 * norm_val)
                g = int(180 + 75 * norm_val)
                r = int(20 + 30 * (1 - norm_val))
            else:
                # Off-diagonal (Misclassifications) -> Soft Slate to Crimson Rose
                b = int(45 + 30 * norm_val)
                g = int(25 + 20 * norm_val)
                r = int(40 + 200 * norm_val)

            cv2.rectangle(img, (x1, y1), (x2, y2), (b, g, r), -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (60, 45, 30), 1)

            # Annotation Text (Count & Percentage)
            count_str = f"{val}"
            pct_str = f"{norm_val*100:.0f}%"

            text_color = (255, 255, 255) if norm_val > 0.4 else (200, 200, 200)

            cv2.putText(
                img,
                count_str,
                (x1 + cell_size // 2 - 12, y1 + cell_size // 2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                text_color,
                1,
                cv2.LINE_AA
            )
            cv2.putText(
                img,
                pct_str,
                (x1 + cell_size // 2 - 14, y1 + cell_size // 2 + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (180, 220, 255) if i == j else (150, 150, 200),
                1,
                cv2.LINE_AA
            )

    # X Axis Labels (Predicted Class)
    for j in range(num_classes):
        x_center = margin_left + j * cell_size + cell_size // 2 - 20
        y_pos = margin_top + num_classes * cell_size + 30
        label_text = class_names[j]
        
        cv2.putText(
            img,
            label_text,
            (x_center - 10, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (203, 213, 225),
            1,
            cv2.LINE_AA
        )

    # Axis Title Labels
    cv2.putText(img, "True Defect Class (Rows)", (15, margin_top - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (56, 189, 248), 1, cv2.LINE_AA)
    cv2.putText(img, "Predicted Defect Class (Columns)", (margin_left + 150, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (56, 189, 248), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), img)
    print(f"[Artifact Generator] Saved high-resolution confusion matrix plot to '{out_path}'")


def generate_confusion_matrix_artifacts(
    all_labels: List[int],
    all_preds: List[int],
    class_names: List[str] = CLASS_NAMES_10,
    png_path: Path = Path("confusion_matrix.png"),
    json_path: Path = Path("confusion_matrix.json")
) -> Dict[str, Any]:
    """Generate raw confusion matrix array, per-class metrics, JSON log, and PNG heatmap plot."""
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    
    # Per-Class Precision, Recall, F1
    per_class_f1 = f1_score(all_labels, all_preds, average=None, labels=list(range(len(class_names))), zero_division=0)
    per_class_prec = precision_score(all_labels, all_preds, average=None, labels=list(range(len(class_names))), zero_division=0)
    per_class_rec = recall_score(all_labels, all_preds, average=None, labels=list(range(len(class_names))), zero_division=0)

    per_class_report = {}
    for idx, name in enumerate(class_names):
        per_class_report[name] = {
            "precision": round(float(per_class_prec[idx]), 4),
            "recall": round(float(per_class_rec[idx]), 4),
            "f1_score": round(float(per_class_f1[idx]), 4),
            "total_samples": int(cm[idx].sum())
        }

    cm_data = {
        "class_names": class_names,
        "confusion_matrix": cm.tolist(),
        "per_class_report": per_class_report,
        "macro_f1": round(float(np.mean(per_class_f1)), 4),
        "overall_accuracy": round(float(np.trace(cm) / np.sum(cm)), 4)
    }

    with open(json_path, "w") as f:
        json.dump(cm_data, f, indent=2)
    print(f"[Artifact Generator] Saved confusion matrix data to '{json_path}'")

    render_confusion_matrix_image(cm, class_names, png_path)

    return cm_data


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_val_f1: float,
    metrics: Dict,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_architecture": "DualFusion-ResNet50-EfficientNet-CBAM",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_f1": best_val_f1,
        "metrics": metrics,
    }
    torch.save(checkpoint, path)
    print(f"  -> Saved best model checkpoint to '{path}' (Val Macro F1: {best_val_f1:.4f})")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==========================================================")
    print(f"   SOTA DUAL FUSION HYBRID WAFER CLASSIFIER TRAINING ENGINE")
    print(f"==========================================================")
    print(f"Execution Device:    {device}")
    print(f"Target Epochs:       {args.epochs}")
    print(f"Batch Size:          {args.batch_size}")
    print(f"Learning Rate:       {args.lr}")
    print(f"Model Type:          {args.model_type}")
    print(f"Use Focal Loss:      {args.use_focal_loss}\n")

    # Build or Load Balanced Augmented Dataset
    samples, stats = build_balanced_samples(
        pkl_path=args.pkl_path,
        target_per_class=args.samples_per_class,
        include_multi_defect=True,
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

    print(f"Train Set: {len(train_subset)} samples | Val Set: {len(val_subset)} samples\n")

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
    last_val_labels: List[int] = []
    last_val_preds: List[int] = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_f1, val_prec, val_rec, val_labels, val_preds = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        last_val_labels = val_labels
        last_val_preds = val_preds

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        epoch_metric = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_macro_f1": round(val_f1, 4),
            "val_precision": round(val_prec, 4),
            "val_recall": round(val_rec, 4),
            "epoch_seconds": round(epoch_time, 2),
            "lr": current_lr,
        }
        history.append(epoch_metric)

        print(
            f"Epoch {epoch:03d}/{args.epochs} [{epoch_time:.1f}s] | "
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

        if epoch % 5 == 0 or epoch == args.epochs:
            with open(args.log_json, "w") as f:
                json.dump({"best_val_f1": best_val_f1, "history": history}, f, indent=2)

    total_duration = time.time() - start_train_time
    print(f"\n==========================================================")
    print(f"Training Complete! Total Duration: {total_duration/60:.2f} mins")
    print(f"Best Validation Macro F1-Score: {best_val_f1:.4f}")
    print(f"==========================================================\n")

    # Generate Final Confusion Matrix & Per-Class Metrics Artifacts
    print("[Artifact Engine] Compiling Confusion Matrix & Per-Class Classification Report...")
    cm_summary = generate_confusion_matrix_artifacts(
        all_labels=last_val_labels,
        all_preds=last_val_preds,
        class_names=CLASS_NAMES_10,
        png_path=args.cm_png,
        json_path=args.cm_json
    )

    print("\n" + "=" * 70)
    print("           PER-CLASS VALIDATION PERFORMANCE METRICS              ")
    print("=" * 70)
    print(f"{'Class Name':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Samples':<8}")
    print("-" * 70)
    for c_name, c_metrics in cm_summary["per_class_report"].items():
        print(f"{c_name:<15} | {c_metrics['precision']*100:.2f}%      | {c_metrics['recall']*100:.2f}%      | {c_metrics['f1_score']*100:.2f}%      | {c_metrics['total_samples']:<8}")
    print("-" * 70)
    print(f"Macro F1-Score:    {cm_summary['macro_f1']*100:.2f}%")
    print(f"Overall Accuracy:  {cm_summary['overall_accuracy']*100:.2f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()
