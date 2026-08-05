"""
Kaggle Dataset Exporter: Generates the Expanded 10-Class Balanced WM-811K Wafer Map Dataset (35,000 Samples).
Creates structured 224x224 PNG images, metadata CSV, pickled datasets, and Kaggle release documentation.
"""

from __future__ import annotations
import os
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from PIL import Image

from pipeline.dataset import build_balanced_samples, EXTENDED_ID_TO_FAILURE_TYPE, NUM_CLASSES_EXTENDED

def export_kaggle_dataset(
    pkl_path: str = "D:/Web Dev/FabMetrics_AI/LSWMD.pkl",
    output_dir: str = "D:/Web Dev/FabMetrics_AI/WM811K_Balanced_Kaggle",
    target_per_class: int = 3500,
    include_multi_defect: bool = True,
    seed: int = 42
):
    out_path = Path(output_dir)
    img_path = out_path / "images"
    out_path.mkdir(parents=True, exist_ok=True)

    img_path.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("      WM-811K EXPANDED 10-CLASS KAGGLE DATASET GENERATOR          ")
    print("==================================================================")
    print(f"Target Samples per Class: {target_per_class} across {NUM_CLASSES_EXTENDED} Classes")
    print(f"Expected Total Samples: {target_per_class * NUM_CLASSES_EXTENDED}")
    print(f"Output Directory: {out_path.resolve()}\n")

    # 1. Build Balanced & Multi-Defect Augmented Samples
    samples, stats = build_balanced_samples(
        pkl_path,
        target_per_class=target_per_class,
        include_multi_defect=include_multi_defect,
        seed=seed
    )

    # Shuffle samples
    random.seed(seed)
    random.shuffle(samples)

    metadata_rows = []
    pkl_records = []

    print(f"\n[Exporter] Saving 224x224 PNG images and compiling metadata.csv for {len(samples)} samples...")

    for idx, (wafer_map, label_id, aug_type) in enumerate(samples, start=1):
        sample_name = f"wafer_sample_{idx:05d}"
        img_filename = f"{sample_name}.png"
        img_full_path = img_path / img_filename
        
        # Colorize wafer map array (0=Background, 1=Normal die, 2=Defect die)
        height, width = wafer_map.shape
        viz_img = np.zeros((height, width, 3), dtype=np.uint8)
        viz_img[wafer_map == 0] = [15, 23, 42]     # Dark slate background
        viz_img[wafer_map == 1] = [30, 58, 95]     # Normal die blue
        viz_img[wafer_map == 2] = [239, 68, 68]    # Defect die vibrant red

        # Resize to standard 224x224 high quality image
        resized_viz = cv2.resize(viz_img, (224, 224), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(img_full_path), resized_viz)

        failure_name = EXTENDED_ID_TO_FAILURE_TYPE[label_id]
        is_aug = 0 if aug_type == "Original" else 1

        metadata_rows.append({
            "sample_id": sample_name,
            "image_filename": f"images/{img_filename}",
            "failure_type": failure_name,
            "label_id": label_id,
            "original_height": height,
            "original_width": width,
            "is_augmented": is_aug,
            "augmentation_method": aug_type
        })

        pkl_records.append({
            "waferMap": wafer_map,
            "failureType": failure_name,
            "label_id": label_id,
            "aug_type": aug_type
        })

        if idx % 5000 == 0 or idx == len(samples):
            print(f" -> Exported {idx}/{len(samples)} images...")

    # 2. Save metadata.csv
    df_meta = pd.DataFrame(metadata_rows)
    csv_file = out_path / "metadata.csv"
    df_meta.to_csv(csv_file, index=False)
    print(f"\n[Exporter] Created {csv_file.name} ({len(df_meta)} rows)")

    # 3. Save pickled dataframe for PyTorch training
    df_pkl = pd.DataFrame(pkl_records)
    pkl_file = out_path / "dataset_balanced.pkl"
    df_pkl.to_pickle(pkl_file)
    print(f"[Exporter] Created {pkl_file.name} ({pkl_file.stat().st_size / (1024*1024):.2f} MB)")

    # 4. Save JSON summary metadata
    summary_file = out_path / "dataset_summary.json"
    with open(summary_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[Exporter] Created {summary_file.name}")

    # 5. Create Kaggle Dataset README.md
    readme_content = f"""# WM-811K 10-Class Balanced & Multi-Defect Wafer Map Dataset ({len(samples)} Samples)

## Overview
This is a **10-class equalized and augmented edition** of the benchmark **WM-811K Wafer Map Dataset** for semiconductor yield analytics and advanced computer vision research.

In the raw dataset (~811,470 wafers), over 80% of wafers are non-defective, and multi-defect patterns are unlabelled. This dataset equalizes all single-defect modes, 'none', and synthetic **Multi-Defect Superimpositions** to **{target_per_class} samples per class** ({len(samples)} total samples).

## Dataset Classes ({NUM_CLASSES_EXTENDED} Total)
1. **Center**: Concentric defect cluster at wafer disk center.
2. **Donut**: Ring-shaped defect formation inside interior die space.
3. **Edge-Loc**: Grouped localized flaw blob hugging the outer perimeter.
4. **Edge-Ring**: Continuous boundary ring of failing dies along extreme edge.
5. **Loc**: Concentrated localized cluster spot.
6. **Random**: Scattered point anomaly distribution across silicon disk.
7. **Scratch**: Linear scratches caused by mechanical transport grippers or slider friction.
8. **Near-full**: Widespread array damage covering majority of wafer surface.
9. **none**: Clean, non-defective wafer substrate.
10. **Multi-Defect**: Complex dual-defect wafer maps (e.g. Scratch + Donut, Edge-Ring + Loc).

## Augmentation Methodology
- Orthogonal Rotations (90°, 180°, 270°) and Flips
- Morphological Dilation & Erosion on defect die masks
- Silicon Die Sensor Read Noise Simulation
- Elastic Lattice Deformation
- Dual-Defect Pattern Superimposition

## Files Included
- `images/`: High-resolution 224x224 RGB colorized PNG previews (`images/wafer_sample_00001.png` ...).
- `metadata.csv`: Table containing image paths, failure labels, label IDs, dimensions, and augmentation methods.
- `dataset_balanced.pkl`: PyTorch / Pandas pickled dataframe containing raw wafer map matrices.
- `dataset_summary.json`: JSON metadata with class distributions.

---
*Created by Chitransh Saxena & Team • Released for Research & Kaggle Publication*
"""
    readme_file = out_path / "README.md"
    with open(readme_file, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"[Exporter] Created {readme_file.name}")

    print("\n==================================================================")
    print("SUCCESS! Expanded 10-Class Kaggle Dataset Export Complete.")
    print(f"Directory: '{out_path.resolve()}' ({len(samples)} total samples)")
    print("==================================================================")

if __name__ == "__main__":
    export_kaggle_dataset()
