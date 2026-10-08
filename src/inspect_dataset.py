#!/usr/bin/env python3
"""
COLONVISION AI — Comprehensive HyperKvasir Dataset Inspection Script

Authoritative Source: image-labels.csv
Dataset Location: C:\\Users\\ganes\\Downloads\\hyper-kvasir-labeled-images\\labeled-images

Performs full inspection across Lower GI and Upper GI tract images:
1. Total images & UUID mapping
2. Exact class list and sample counts from image-labels.csv
3. Image format and resolution analysis
4. Corrupted file detection
5. Duplicate file detection via MD5 cryptographic hashing
6. Target class verification (Colorectal Cancer vs. Diverticular Disease)
7. Generates results/dataset_report.csv, results/dataset_report.json, and results/figures/class_distribution.png
"""

import os
import sys
import json
import glob
import hashlib
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


CANCER_KEYWORDS = ["cancer", "carcinoma", "adenocarcinoma", "malignan"]
DIVERTICULAR_KEYWORDS = ["diverticul", "diverticula", "diverticulosis", "diverticulitis"]


def compute_file_hash(filepath: Path, chunk_size: int = 65536) -> str:
    """Computes MD5 hash of a file for duplicate detection."""
    md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(chunk_size):
                md5.update(chunk)
        return md5.hexdigest()
    except Exception:
        return ""


def run_inspection(dataset_dir: str, output_dir: str, figures_dir: str):
    dataset_path = Path(dataset_dir)
    results_path = Path(output_dir)
    figs_path = Path(figures_dir)

    csv_path = dataset_path / "image-labels.csv"
    if not csv_path.exists():
        # Search recursively for image-labels.csv
        found_csvs = list(dataset_path.glob("**/image-labels.csv"))
        if found_csvs:
            csv_path = found_csvs[0]
        else:
            print(f"[X] Error: image-labels.csv not found under {dataset_dir}")
            return

    print("=" * 75)
    print("COLONVISION AI - DATASET INSPECTION REPORT")
    print("=" * 75)
    print(f"Authoritative Metadata File: {csv_path}")

    # Read CSV
    df = pd.read_csv(csv_path)
    total_csv_records = len(df)
    print(f"Total metadata records in CSV: {total_csv_records}")

    # Map image files on disk
    image_files = list(dataset_path.glob("**/*.jpg")) + list(dataset_path.glob("**/*.png")) + list(dataset_path.glob("**/*.jpeg"))
    uuid_to_path = {p.stem: p for p in image_files}
    print(f"Total image files found on disk: {len(image_files)}")

    # Class distribution analysis
    finding_counts = df['Finding'].value_counts()
    organ_counts = df['Organ'].value_counts()
    classification_counts = df['Classification'].value_counts()

    # Lower vs Upper GI findings
    lower_gi_df = df[df['Organ'] == 'Lower GI']
    upper_gi_df = df[df['Organ'] == 'Upper GI']

    lower_gi_findings = lower_gi_df['Finding'].value_counts().to_dict()
    upper_gi_findings = upper_gi_df['Finding'].value_counts().to_dict()

    # Image integrity, resolution, duplicate check
    corrupted_files = []
    file_hashes = defaultdict(list)
    dimensions = []
    formats = Counter()

    for idx, row in df.iterrows():
        v_id = row['Video file']
        img_path = uuid_to_path.get(v_id)
        if not img_path or not img_path.exists():
            corrupted_files.append({"video_file": v_id, "error": "File not found on disk"})
            continue

        try:
            with Image.open(img_path) as img:
                fmt = img.format or "JPEG"
                w, h = img.size
                formats[fmt] += 1
                dimensions.append((w, h))
                img.verify()
        except Exception as e:
            corrupted_files.append({"video_file": v_id, "filepath": str(img_path), "error": str(e)})
            continue

        md5_hash = compute_file_hash(img_path)
        if md5_hash:
            file_hashes[md5_hash].append(v_id)

    duplicates = {h: ids for h, ids in file_hashes.items() if len(ids) > 1}
    total_duplicate_images = sum(len(ids) - 1 for ids in duplicates.values())

    # Build Class Breakdown
    class_records = []
    for finding, count in finding_counts.items():
        pct = round((count / total_csv_records) * 100, 2)
        
        # Check target class matching
        f_lower = finding.lower().replace("-", " ")
        is_cancer = any(k in f_lower for k in CANCER_KEYWORDS)
        is_diverticular = any(k in f_lower for k in DIVERTICULAR_KEYWORDS)

        organ = df[df['Finding'] == finding]['Organ'].iloc[0]
        classification = df[df['Finding'] == finding]['Classification'].iloc[0]

        class_records.append({
            "original_label": finding,
            "sample_count": count,
            "percentage": pct,
            "organ": organ,
            "classification": classification,
            "is_cancer_candidate": is_cancer,
            "is_diverticular_candidate": is_diverticular
        })

    report_df = pd.DataFrame(class_records)

    # Resolution statistics
    if dimensions:
        widths, heights = zip(*dimensions)
        res_stats = {
            "min_resolution": f"{min(widths)}x{min(heights)}",
            "max_resolution": f"{max(widths)}x{max(heights)}",
            "mean_resolution": f"{int(np.mean(widths))}x{int(np.mean(heights))}",
            "most_common_resolutions": [f"{w}x{h} ({cnt} images)" for (w, h), cnt in Counter(dimensions).most_common(5)]
        }
    else:
        res_stats = {}

    counts_list = [r["sample_count"] for r in class_records]
    max_c = max(counts_list)
    min_c = min(counts_list)
    imbalance_ratio = round(max_c / min_c, 2) if min_c > 0 else float("inf")

    # Target class verification
    cancer_classes = [r for r in class_records if r["is_cancer_candidate"]]
    diverticular_classes = [r for r in class_records if r["is_diverticular_candidate"]]

    # Save outputs
    results_path.mkdir(parents=True, exist_ok=True)
    figs_path.mkdir(parents=True, exist_ok=True)

    csv_out = results_path / "dataset_report.csv"
    json_out = results_path / "dataset_report.json"
    plot_out = figs_path / "class_distribution.png"

    report_df.to_csv(csv_out, index=False)

    report_json = {
        "dataset_name": "HyperKvasir Labeled Images Dataset",
        "authoritative_source": str(csv_path),
        "total_images_in_csv": total_csv_records,
        "total_images_on_disk": len(image_files),
        "corrupted_images_count": len(corrupted_files),
        "corrupted_files": corrupted_files,
        "duplicate_groups_count": len(duplicates),
        "duplicate_images_count": total_duplicate_images,
        "total_classes": len(class_records),
        "organ_distribution": organ_counts.to_dict(),
        "classification_category_distribution": classification_counts.to_dict(),
        "lower_gi_findings": lower_gi_findings,
        "upper_gi_findings": upper_gi_findings,
        "image_formats": dict(formats),
        "resolution_statistics": res_stats,
        "class_imbalance_ratio_max_to_min": f"{imbalance_ratio}:1 ({max_c}:{min_c})",
        "cancer_classes_detected": cancer_classes,
        "diverticular_classes_detected": diverticular_classes,
        "class_breakdown": class_records
    }

    with open(json_out, "w") as f:
        json.dump(report_json, f, indent=2)

    print(f"\n[OK] CSV report saved to: {csv_out}")
    print(f"[OK] JSON report saved to: {json_out}")

    # Plot
    plt.figure(figsize=(14, 7))
    if HAS_SEABORN:
        sns.barplot(data=report_df, x="original_label", y="sample_count", palette="viridis")
    else:
        plt.bar(report_df["original_label"], report_df["sample_count"], color="teal")
    plt.title("HyperKvasir Complete Class Distribution (Authoritative image-labels.csv)", fontsize=14, fontweight="bold")
    plt.xlabel("Original Class Label (Finding)", fontsize=11)
    plt.ylabel("Number of Images", fontsize=11)
    plt.xticks(rotation=60, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(plot_out, dpi=300)
    plt.close()

    print(f"[OK] Distribution plot saved to: {plot_out}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect HyperKvasir Dataset for ColonVision AI")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=r"C:\Users\ganes\Downloads\hyper-kvasir-labeled-images",
        help="Path to downloaded hyper-kvasir-labeled-images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for reports"
    )
    parser.add_argument(
        "--figures_dir",
        type=str,
        default="results/figures",
        help="Output directory for figures"
    )
    args = parser.parse_args()

    run_inspection(args.dataset_dir, args.output_dir, args.figures_dir)
