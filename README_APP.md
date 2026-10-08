# ColonVision AI — Web Application Guide

## Overview

**ColonVision AI** is an explainable deep learning demonstration application for the differential classification of **Colorectal Cancer** and **Colon Diverticula** from colonoscopy images. The system combines transfer-learning classification via **ResNet-50** with visual interpretability through **Grad-CAM (Gradient-weighted Class Activation Mapping)**.

> [!WARNING]
> **Research Prototype Only:** This system is an academic research demonstration and is **not** a certified medical diagnostic device. It must not be used as a substitute for professional medical evaluation, endoscopic assessment, or biopsy.

---

## 1. Project Architecture

```
ColonVision-AI/
├── app/
│   └── app.py                     # Streamlit web application
├── data/
│   └── processed/
│       ├── train/                 # 97 Cancer, 20 Diverticula
│       ├── val/                   # 21 Cancer, 4 Diverticula
│       └── test/                  # 21 Cancer, 5 Diverticula
├── models/
│   ├── resnet50_best.pth          # Best ResNet-50 model weights (94.4 MB)
│   ├── densenet121_best.pth       # DenseNet-121 model weights (28.4 MB)
│   ├── efficientnet_b0_best.pth   # EfficientNet-B0 model weights (16.3 MB)
│   └── endoscopy_validator_profile.json # Calibrated OOD protection profile
├── results/
│   ├── gradcam/                   # Grad-CAM visualizations and summary CSV
│   ├── GRADCAM_REPORT.md          # Detailed explainability report
│   ├── PHASE_5_MODEL_TRAINING_REPORT.md  # Training comparison report
│   ├── PHASE_8_VALIDATOR_REPORT.md # OOD validation and calibration report
│   └── model_comparison.csv       # Comparative metrics table
├── src/
│   ├── prepare_dataset.py         # Stratified 70/15/15 dataset builder
│   ├── train_models.py            # Model training and evaluation pipeline
│   ├── explain_gradcam.py         # Batch Grad-CAM generation script
│   └── endoscopy_validator.py     # Endoscopy-domain validation & OOD protection
├── tests/
│   ├── test_app_inference.py      # App pipeline unit test
│   └── test_endoscopy_validator.py# OOD gatekeeper test suite
├── requirements.txt               # Python package dependencies
├── README_APP.md                  # Application guide (this file)
└── README.md                      # Project root documentation
```

---

## 2. Model Information

- **Architecture:** ResNet-50 (Deep Residual Network with Bottleneck blocks)
- **Pretraining:** ImageNet-1K (`IMAGENET1K_V1`)
- **Model Checkpoint:** `models/resnet50_best.pth`
- **Class Mapping:**
  - `Class 0`: `colon_diverticula` (Benign outpouching)
  - `Class 1`: `colorectal_cancer` (Malignant neoplasm)
- **Target Explainability Layer:** `model.layer4[-1]` (Final residual bottleneck block, 2048 channels)
- **Preprocessing Pipeline (Deterministic):**
  - Input frame resized to $256 \times 256$
  - Center-cropped to $224 \times 224$
  - Normalized with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`

---

## 3. Installation

Ensure Python 3.10+ is installed in your virtual environment:

```bash
# Navigate to the project root directory
cd ColonVision-AI

# Activate your virtual environment (if applicable)
# On Windows PowerShell:
.venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

---

## 4. How to Run the Application

Launch the Streamlit web server locally:

```bash
# Standard command:
streamlit run app/app.py

# Or via Python module directly (recommended if streamlit CLI is not in PATH):
python -m streamlit run app/app.py
```

Once started, the application will be accessible at:
```
Local URL: http://localhost:8501
```

---

## 5. Application Features

1. **Dual Input Methods:**
   - **File Upload:** Upload any standard endoscopic image (`.jpg`, `.jpeg`, `.png`).
   - **Demonstration Samples:** Quick-load representative test benchmark samples from the sidebar.
2. **Deterministic Inference:**
   - Evaluates the frame using the exact deterministic evaluation pipeline.
   - Computes raw logits, softmax class probabilities, and prediction confidence.
3. **Interactive Grad-CAM Explainability:**
   - Extracts gradient activations directly from `model.layer4[-1]`.
   - Displays a 3-panel comparative layout:
     1. Preprocessed Input Crop ($224 \times 224$)
     2. Grad-CAM Activation Heatmap (Jet Colormap)
     3. Model Attention Overlay with user-adjustable alpha transparency slider.
4. **Diagnostic Metrics & Diagnostics:**
   - Metric cards showing prediction confidence and decision margin.
   - Expandable technical diagnostic panel with raw tensor outputs and shapes.
5. **Privacy Safeguard:**
   - Uploaded frames are processed in-memory and are never stored to permanent disk.

---

## 6. Clinical & Engineering Limitations

1. **Extreme Minority Sample Size:** The underlying dataset contains only 29 colon diverticula images in total (20 train, 4 val, 5 test). While the ResNet-50 model achieved 100% test accuracy on this split, minority class representation is constrained.
2. **Coarse Spatial Attribution:** The feature map at `model.layer4[-1]` has a spatial resolution of $7 \times 7$ pixels. Upsampling to $224 \times 224$ produces smooth localization, but should not be taken as exact lesion borders.
3. **Non-Pathological Artifacts:** Endoscopic light glints (specular reflections), fluid bubbles, and folds can occasionally draw gradient attention.
4. **No Histological Replacement:** Visual heatmaps are purely interpretability aids and do not provide biopsy-confirmed histological assessment.
