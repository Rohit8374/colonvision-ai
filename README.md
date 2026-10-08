# COLONVISION AI

> **An Explainable Deep Learning Framework for Differential Classification of Colorectal Cancer and Diverticular Disease from Colonoscopy Images**

---

### ⚠️ Research Disclaimer
This repository contains an **academic medical-image research project**. The final system is designed and evaluated strictly as an **AI-assisted research/decision-support prototype** and **NOT as a replacement for professional medical diagnosis or clinical evaluation**.

---

## 📌 Project Overview & Objectives

The primary objective of **ColonVision AI** is to investigate whether state-of-the-art deep learning architectures can accurately and explainably differentiate between **Colorectal Cancer** and **Diverticular Disease / Diverticula** using lower GI tract colonoscopy/endoscopy images.

### Key Framework Components:
- **Architectures Under Investigation**:
  - ResNet (e.g., ResNet-50)
  - DenseNet (e.g., DenseNet-121)
  - EfficientNet (e.g., EfficientNet-B0 / B4)
- **Model Explainability**:
  - Grad-CAM (Gradient-weighted Class Activation Mapping) for visual localization of relevant endoscopic features
- **Evaluation Metrics**:
  - Accuracy, Precision, Recall, F1-Score, ROC-AUC Score, Confusion Matrix

---

## 📁 Repository Structure

```text
ColonVision-AI/
├── data/
│   ├── raw/          # Original dataset (Read-Only raw images & metadata)
│   ├── processed/    # Processed / standard-size images
│   └── splits/       # Train / Validation / Test split manifests
├── notebooks/        # Exploratory analysis & experiments
├── src/              # Source code modules (inspection, preprocessing, training, Grad-CAM)
│   └── inspect_dataset.py
├── models/           # Trained model checkpoints (.pt, .pth) [git-ignored]
├── results/          # Output analysis reports & figures
│   ├── figures/      # Class distribution & training graphs
│   ├── metrics/      # CSV/JSON performance evaluation metrics
│   └── gradcam/      # Grad-CAM visual heatmaps
├── app/              # Decision-support web interface
├── requirements.txt  # Project dependencies
├── README.md         # Project documentation
└── .gitignore        # Git exclusion rules
```

---

## 🔒 Dataset Integrity & Guidelines

1. **Read-Only Raw Data**: The contents of `data/raw/` are treated as strictly read-only.
2. **Strict Medical Label Honesty**: Original dataset labels, metadata, and folder structures are preserved without reinterpretation or fabrication.
3. **No Unvalidated Equivalencies**: Polyps, hemorrhoids, normal mucosa, or general inflammation are NOT treated as Colorectal Cancer or Diverticular Disease unless explicitly labeled by the source dataset.
4. **Git Safety**: `data/raw/`, `data/processed/`, and model weights are excluded via `.gitignore` to protect dataset license rights and repository size.

---

## 🚀 Getting Started

### 1. Environment Setup
Create and activate the virtual environment, then install dependencies:

```bash
# Navigate to project directory
cd ColonVision-AI

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install required packages
pip install -r requirements.txt
```

### 2. Dataset Placement
Download and extract the **HyperKvasir** dataset into:
```text
ColonVision-AI/data/raw/
```
Ensure original metadata files (e.g., `image_metadata.csv` or folder hierarchies) remain inside `data/raw/`.

### 3. Dataset Inspection & Verification
Before proceeding with preprocessing or model training, run the dataset inspection tool:

```bash
python src/inspect_dataset.py --data_dir data/raw
```

This script will verify image readability, report exact original class counts, evaluate class imbalance, check target class availability, save reports to `results/dataset_report.csv` & `results/dataset_report.json`, and output the distribution plot to `results/figures/class_distribution.png`.
