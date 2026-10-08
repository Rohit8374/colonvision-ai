#!/usr/bin/env python3
"""
COLONVISION AI - Dataset Preparation Script
============================================
Copies ONLY the two target classes from GastroVision:
  - Colorectal cancer  (139 images)
  - Colon diverticula  (29 images)

Steps performed:
  1. Verify source images via MD5 hashing, detect known duplicate
  2. Copy images read-only into data/raw/ (original untouched)
  3. Stratified 70/15/15 split with fixed random seed
  4. Copy split images into data/processed/train|val|test/
  5. Generate CSV report, JSON summary, and distribution chart

Rules enforced:
  - No augmentation at this stage
  - No resizing
  - No label reinterpretation
  - No original files modified or deleted
  - No model training
"""

import os
import sys
import json
import shutil
import hashlib
import argparse
import csv
from pathlib import Path
from collections import defaultdict
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ── fixed seed for full reproducibility ──────────────────────────────────────
RANDOM_SEED = 42

# ── source class directories inside GastroVision ─────────────────────────────
GASTROVISION_ROOT = r"C:\Users\ganes\Downloads\GastroVision\Gastrovision"
SOURCE_CLASSES = {
    "colorectal_cancer": "Colorectal cancer",
    "colon_diverticula": "Colon diverticula",
}

# ── project paths (relative to project root) ─────────────────────────────────
RAW_ROOT       = Path("data/raw")
PROCESSED_ROOT = Path("data/processed")
RESULTS_DIR    = Path("results")
FIGURES_DIR    = Path("results/figures")

# ── split ratios ─────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def md5_of(filepath: Path, chunk=65536) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while buf := f.read(chunk):
            h.update(buf)
    return h.hexdigest()


def collect_unique_images(src_dir: Path, class_label: str):
    """
    Collect all .jpg/.png images from src_dir.
    Detect exact MD5 duplicates: keep the FIRST occurrence alphabetically, discard rest.
    Returns list of dicts with keys: path, filename, md5, class_label
    """
    valid_exts = {".jpg", ".jpeg", ".png"}
    candidates = sorted(
        [p for p in src_dir.iterdir() if p.suffix.lower() in valid_exts],
        key=lambda p: p.name,
    )

    seen_hashes = {}      # md5 -> first path seen
    duplicates_removed = []
    unique_images = []

    for p in candidates:
        h = md5_of(p)
        if h in seen_hashes:
            duplicates_removed.append({
                "kept": str(seen_hashes[h]),
                "discarded": str(p),
                "md5": h,
            })
        else:
            seen_hashes[h] = p
            unique_images.append({
                "path": p,
                "filename": p.name,
                "md5": h,
                "class_label": class_label,
            })

    return unique_images, duplicates_removed


def split_images(images: list, train_r: float, val_r: float, seed: int):
    """
    Stratified (within-class) random split into train / val / test.
    No image appears in more than one split.
    """
    rng = random.Random(seed)
    shuffled = images[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * train_r)
    n_val   = round(n * val_r)

    train = shuffled[:n_train]
    val   = shuffled[n_train : n_train + n_val]
    test  = shuffled[n_train + n_val :]

    return train, val, test


def copy_images(image_list: list, dest_dir: Path):
    """Copy images to dest_dir. Read-only: never touches source."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for img in image_list:
        src = img["path"]
        dst = dest_dir / img["filename"]
        # Ensure no filename collision (keep original name)
        shutil.copy2(src, dst)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("COLONVISION AI - Dataset Preparation")
    print("=" * 70)
    print(f"Random Seed : {RANDOM_SEED}")
    print(f"Split Ratio : {TRAIN_RATIO:.0%} / {VAL_RATIO:.0%} / {TEST_RATIO:.0%}\n")

    all_splits = {}           # class_label -> {train, val, test}
    all_duplicates = []
    original_counts = {}

    # ── Step 1: Collect & de-duplicate each class ─────────────────────────
    for label, folder_name in SOURCE_CLASSES.items():
        src = Path(GASTROVISION_ROOT) / folder_name
        if not src.exists():
            print(f"[ERROR] Source folder not found: {src}")
            sys.exit(1)

        unique_imgs, dups = collect_unique_images(src, label)
        all_duplicates.extend(dups)
        original_counts[label] = len(unique_imgs) + len(dups)  # before de-dup

        if dups:
            print(f"[!] Duplicate detected in '{label}':")
            for d in dups:
                print(f"    KEPT     : {d['kept']}")
                print(f"    DISCARDED: {d['discarded']}")
                print(f"    MD5      : {d['md5']}")

        print(f"Class '{label}': {len(unique_imgs)} unique images "
              f"(original count incl. duplicates: {original_counts[label]})")

        # ── Step 2: Copy all unique images to data/raw/ ───────────────────
        raw_class_dir = RAW_ROOT / label
        raw_class_dir.mkdir(parents=True, exist_ok=True)
        for img in unique_imgs:
            dst = raw_class_dir / img["filename"]
            if not dst.exists():          # never overwrite
                shutil.copy2(img["path"], dst)

        # ── Step 3: Split ─────────────────────────────────────────────────
        train, val, test = split_images(unique_imgs, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED)
        all_splits[label] = {"train": train, "val": val, "test": test}

        print(f"  -> train: {len(train)} | val: {len(val)} | test: {len(test)}")

    # ── Step 4: Verify zero cross-split leakage (MD5 check) ───────────────
    print("\nVerifying data-leakage safety (cross-split hash check)...")
    for label in all_splits:
        splits = all_splits[label]
        train_hashes = {img["md5"] for img in splits["train"]}
        val_hashes   = {img["md5"] for img in splits["val"]}
        test_hashes  = {img["md5"] for img in splits["test"]}

        leakage = (train_hashes & val_hashes) | (train_hashes & test_hashes) | (val_hashes & test_hashes)
        if leakage:
            print(f"[CRITICAL] Data leakage detected in class '{label}'! Hashes: {leakage}")
            sys.exit(1)
        else:
            print(f"  [{label}] No leakage detected. Splits are disjoint.")

    # ── Step 5: Copy splits to data/processed/ ────────────────────────────
    print("\nCopying splits to data/processed/ ...")
    for label, splits in all_splits.items():
        for split_name, imgs in splits.items():
            dest = PROCESSED_ROOT / split_name / label
            copy_images(imgs, dest)
            print(f"  Copied {len(imgs):3d} images -> {dest}")

    # ── Step 6: Generate CSV report ───────────────────────────────────────
    csv_rows = []
    total_images = sum(
        len(imgs)
        for splits in all_splits.values()
        for imgs in splits.values()
    )

    for split_name in ["train", "val", "test"]:
        for label in SOURCE_CLASSES:
            cnt = len(all_splits[label][split_name])
            pct = round(cnt / total_images * 100, 2)
            csv_rows.append({
                "split": split_name,
                "class": label,
                "image_count": cnt,
                "percentage": pct,
            })

    csv_path = RESULTS_DIR / "split_report.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "class", "image_count", "percentage"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n[OK] Split report saved: {csv_path}")

    # ── Step 7: Generate JSON summary ─────────────────────────────────────
    summary = {
        "random_seed": RANDOM_SEED,
        "split_ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "total_unique_images_used": sum(
            len(unique_imgs)
            for label in all_splits
            for unique_imgs in [all_splits[label]["train"] + all_splits[label]["val"] + all_splits[label]["test"]]
        ),
        "classes": {},
        "duplicate_handling": {
            "policy": "Exact MD5 duplicates removed before splitting. Duplicate may not appear in any split.",
            "duplicates_found": all_duplicates,
        },
        "data_leakage_check": "PASSED - all splits verified disjoint by MD5 hash",
        "augmentation_policy": "No augmentation applied at this stage. Augmentation to be applied to training set ONLY during model training.",
        "preprocessing_applied": "None. Raw images copied as-is.",
    }

    for label in SOURCE_CLASSES:
        splits = all_splits[label]
        summary["classes"][label] = {
            "original_count_incl_duplicates": original_counts[label],
            "unique_images_used": len(splits["train"]) + len(splits["val"]) + len(splits["test"]),
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        }

    json_path = RESULTS_DIR / "split_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] JSON summary saved: {json_path}")

    # ── Step 8: Generate distribution chart ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("ColonVision AI - Dataset Split Distribution", fontsize=14, fontweight="bold")

    split_names = ["train", "val", "test"]
    classes = list(SOURCE_CLASSES.keys())
    colors = {"colorectal_cancer": "#e05c5c", "colon_diverticula": "#5c8ae0"}
    x = np.arange(len(split_names))
    width = 0.35

    for i, label in enumerate(classes):
        counts = [len(all_splits[label][s]) for s in split_names]
        axes[0].bar(x + (i - 0.5) * width, counts, width, label=label.replace("_", " ").title(), color=colors[label])

    axes[0].set_xlabel("Split")
    axes[0].set_ylabel("Image Count")
    axes[0].set_title("Image Count per Split & Class")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["Train", "Val", "Test"])
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.4)

    # Pie chart for overall class distribution
    overall_counts = [sum(len(all_splits[l][s]) for s in split_names) for l in classes]
    axes[1].pie(
        overall_counts,
        labels=[l.replace("_", " ").title() for l in classes],
        autopct="%1.1f%%",
        colors=[colors[l] for l in classes],
        startangle=90,
    )
    axes[1].set_title("Overall Class Distribution")

    plt.tight_layout()
    chart_path = FIGURES_DIR / "split_distribution.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[OK] Distribution chart saved: {chart_path}")

    # ── Final Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 70)
    print(f"{'Class':<25} {'Original':>10} {'Train':>8} {'Val':>6} {'Test':>6}")
    print("-" * 60)
    for label in SOURCE_CLASSES:
        splits = all_splits[label]
        print(f"{label:<25} {original_counts[label]:>10} "
              f"{len(splits['train']):>8} {len(splits['val']):>6} {len(splits['test']):>6}")
    print("-" * 60)
    total_train = sum(len(all_splits[l]["train"]) for l in SOURCE_CLASSES)
    total_val   = sum(len(all_splits[l]["val"])   for l in SOURCE_CLASSES)
    total_test  = sum(len(all_splits[l]["test"])  for l in SOURCE_CLASSES)
    grand_total = total_train + total_val + total_test
    print(f"{'TOTAL':<25} {sum(original_counts.values()):>10} "
          f"{total_train:>8} {total_val:>6} {total_test:>6}")
    print(f"\nGrand total unique images used: {grand_total}")
    print(f"Duplicate images excluded     : {len(all_duplicates)}")
    print(f"Random seed                   : {RANDOM_SEED}")
    print(f"\nNOTE: No model training initiated. Awaiting user approval.")
    print("=" * 70)


if __name__ == "__main__":
    main()
