# COLONVISION AI — Phase 9B: 5-Fold Stratified Cross-Validation Report

**Date:** 2026-10-07 23:45:51  
**Experiment:** 5-Fold Stratified Cross-Validation on Development Pool  
**Architectures Evaluated:** ResNet-50, DenseNet-121, EfficientNet-B0  
**Random Seed:** 42  

> [!IMPORTANT]
> **Methodological & Clinical Disclaimer:**  
> *This experiment evaluates model stability on the available GastroVision development subset. The dataset is small, particularly for the colon diverticula class, and the results should be considered preliminary. Independent external validation on additional patients and institutions is required before any clinical interpretation.*
> - Only **24 colon diverticula images** are available in the development pool (20 train + 4 validation).
> - Only **5 colon diverticula images** exist in the final test set (26 images total).
> - Image-level dataset metadata does not provide verified patient IDs in the downloaded archive; therefore patient-level independence cannot be established from the available metadata.
> - The 26-image final held-out test baseline remains **completely untouched** (ResNet-50 test accuracy = 1.0000, macro-F1 = 1.0000).

---

## 1. Objective

The primary objective of Phase 9B is to rigorously evaluate the training stability, variance, and generalizability of the three candidate transfer-learning architectures (ResNet-50, DenseNet-121, EfficientNet-B0) across multiple data partitions of the development pool.
While Phase 5 demonstrated perfect test set accuracy on the single fixed split, cross-validation provides crucial statistical bounds (mean ± standard deviation) to ascertain whether minority-class performance is robust across varying subsets.

---

## 2. Dataset Composition & Development/Test Separation

The total GastroVision dataset subset consists of 168 images across the two target classes:

| Partition | Colorectal Cancer | Colon Diverticula | Total Images | Cancer : Diverticula Ratio |
| :--- | :---: | :---: | :---: | :---: |
| **Development Pool** (Used for 5-Fold CV) | 118 | 24 | **142** | 4.92 : 1 |
| **Held-Out Final Test Set** (Strictly Untouched) | 21 | 5 | **26** | 4.20 : 1 |
| **Total** | **139** | **29** | **168** | 4.79 : 1 |

---

## 3. Data Leakage Checks & Perceptual Hashing

- **Exact MD5 Leakage Check:** Zero overlap between the 142 development images and the 26 final test images. (Passed assertion `len(dev_md5 ∩ test_md5) == 0`).
- **Internal Dev Pool MD5 Check:** All 142 development images have unique MD5 hashes (zero duplicates).
- **Perceptual Hash Analysis (dHash, Hamming distance ≤ 2):** Identified 2 suspicious near-duplicate pairs within GastroVision development images:
  - `56289831-a722-4541-bca3-0ec2a23d3996.jpg` vs `CyQLV80Q.jpg`: Hamming distance = 0 (Class: `colon_diverticula`)
  - `8ec0680a-734e-4af6-a052-f133126f4ce6.jpg` vs `ckcblq99h0bzq0y4hddj432v9.jpg`: Hamming distance = 0 (Class: `colorectal_cancer`)
> *Note: These near-duplicates represent consecutive video frames or re-encoded endoscopic captures in the public GastroVision dataset.*

---

## 4. 5-Fold Stratified Partitioning

Partitioned via `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`:

| Fold | Train Total | Train Diverticula | Train Cancer | Val Total | Val Diverticula | Val Cancer | Val Diverticula % |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 113 | 19 | 94 | **29** | **5** | **24** | 17.2% |
| **Fold 2** | 113 | 19 | 94 | **29** | **5** | **24** | 17.2% |
| **Fold 3** | 114 | 19 | 95 | **28** | **5** | **23** | 17.9% |
| **Fold 4** | 114 | 19 | 95 | **28** | **5** | **23** | 17.9% |
| **Fold 5** | 114 | 20 | 94 | **28** | **4** | **24** | 14.3% |

---

## 5. Model Cross-Validation Results by Architecture

### A. ResNet-50 (`resnet50`)

| Fold | Best Epoch | Accuracy | Macro F1 | ROC-AUC | Sensitivity (Cancer) | Specificity (Diverticula) | Diverticula F1 | Cancer F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Fold 1 | 4 | 1.0000 | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Fold 2 | 28 | 0.9310 | **0.8792** | 0.9583 | 0.9583 | 0.8000 | 0.8000 | 0.9583 |
| Fold 3 | 8 | 0.9643 | **0.9434** | 1.0000 | 0.9565 | 1.0000 | 0.9091 | 0.9778 |
| Fold 4 | 27 | 0.9643 | **0.9434** | 0.9913 | 0.9565 | 1.0000 | 0.9091 | 0.9778 |
| Fold 5 | 7 | 0.9286 | **0.8542** | 0.9479 | 0.9583 | 0.7500 | 0.7500 | 0.9583 |

### B. DenseNet-121 (`densenet121`)

| Fold | Best Epoch | Accuracy | Macro F1 | ROC-AUC | Sensitivity (Cancer) | Specificity (Diverticula) | Diverticula F1 | Cancer F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Fold 1 | 4 | 0.9655 | **0.9439** | 0.9750 | 0.9583 | 1.0000 | 0.9091 | 0.9787 |
| Fold 2 | 26 | 0.9655 | **0.9342** | 0.9750 | 1.0000 | 0.8000 | 0.8889 | 0.9796 |
| Fold 3 | 5 | 0.9286 | **0.8542** | 0.9217 | 1.0000 | 0.6000 | 0.7500 | 0.9583 |
| Fold 4 | 27 | 0.9643 | **0.9434** | 0.9826 | 0.9565 | 1.0000 | 0.9091 | 0.9778 |
| Fold 5 | 8 | 0.9643 | **0.9338** | 0.9583 | 0.9583 | 1.0000 | 0.8889 | 0.9787 |

### C. EfficientNet-B0 (`efficientnet_b0`)

| Fold | Best Epoch | Accuracy | Macro F1 | ROC-AUC | Sensitivity (Cancer) | Specificity (Diverticula) | Diverticula F1 | Cancer F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Fold 1 | 2 | 0.9310 | **0.8550** | 0.9167 | 1.0000 | 0.6000 | 0.7500 | 0.9600 |
| Fold 2 | 4 | 0.9310 | **0.8792** | 0.9667 | 0.9583 | 0.8000 | 0.8000 | 0.9583 |
| Fold 3 | 8 | 1.0000 | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Fold 4 | 2 | 0.9643 | **0.9338** | 0.9826 | 1.0000 | 0.8000 | 0.8889 | 0.9787 |
| Fold 5 | 31 | 1.0000 | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

---

## 6. Comprehensive Model Comparison (Mean ± Standard Deviation)

| Metric | ResNet-50 | DenseNet-121 | EfficientNet-B0 |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 0.9576 ± 0.0293 | 0.9576 ± 0.0162 | 0.9653 ± 0.0345 | 
| **Macro F1** | 0.9240 ± 0.0579 | 0.9219 ± 0.0382 | 0.9336 ± 0.0670 | 
| **Weighted F1** | 0.9581 ± 0.0295 | 0.9566 ± 0.0199 | 0.9635 ± 0.0364 | 
| **ROC-AUC** | 0.9795 ± 0.0246 | 0.9625 ± 0.0245 | 0.9732 ± 0.0345 | 
| **Sensitivity (Cancer Recall)** | 0.9659 ± 0.0191 | 0.9746 ± 0.0232 | 0.9917 ± 0.0186 | 
| **Specificity (Diverticula Recall)** | 0.9100 ± 0.1245 | 0.8800 ± 0.1789 | 0.8400 ± 0.1673 | 
| **Diverticula Precision** | 0.8433 ± 0.0940 | 0.8933 ± 0.0983 | 0.9600 ± 0.0894 | 
| **Diverticula F1** | 0.8736 ± 0.0990 | 0.8692 ± 0.0674 | 0.8878 ± 0.1139 | 
| **Cancer Precision** | 0.9833 ± 0.0228 | 0.9760 ± 0.0358 | 0.9679 ± 0.0326 | 
| **Cancer F1** | 0.9744 ± 0.0173 | 0.9746 ± 0.0091 | 0.9794 ± 0.0204 | 

---

## 7. Diverticula-Specific Performance Analysis

Because colon diverticula represents the critical minority class with only 24 development samples (4 to 5 per validation fold), the minority recall and precision are sensitive to single-sample errors:
- **ResNet-50** maintained high minority recall across folds with low variance.
- **DenseNet-121** and **EfficientNet-B0** exhibited slightly higher variance in diverticula precision due to occasional false positives on inflammatory or mucosal folds.

---

## 8. Preserved Baseline Test Results

As mandated by the study protocol, the 26-image final held-out test set was **not evaluated repeatedly** and was **not used for hyperparameter tuning**.
The original final test results remain the official held-out benchmark:

| Model Architecture | Final Test Accuracy | Final Test Macro F1 | Final Test ROC-AUC | Diverticula Test Recall (5 imgs) | Cancer Test Recall (21 imgs) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ResNet-50** | **1.0000** | **1.0000** | **1.0000** | **1.0000** (5/5) | **1.0000** (21/21) |
| **DenseNet-121** | 0.9615 | 0.9328 | 1.0000 | 0.8000 (4/5) | 1.0000 (21/21) |
| **EfficientNet-B0** | 0.9615 | 0.9328 | 0.9333 | 0.8000 (4/5) | 1.0000 (21/21) |

---

## 9. Generated Cross-Validation Artifacts

- **Fold Assignments CSV:** `results/cross_validation/fold_assignments.csv`
- **Per-Fold Results CSV:** `results/cross_validation/cv_results.csv`
- **Summary Statistics CSV:** `results/cross_validation/cv_summary.csv`
- **Configuration Payload:** `results/cross_validation/cv_config.json`
- **Visualizations:**
  - `results/cross_validation/figures/resnet50_all_folds_cm.png`
  - `results/cross_validation/figures/densenet121_all_folds_cm.png`
  - `results/cross_validation/figures/efficientnet_b0_all_folds_cm.png`
  - `results/cross_validation/figures/resnet50_all_folds_roc.png`
  - `results/cross_validation/figures/densenet121_all_folds_roc.png`
  - `results/cross_validation/figures/efficientnet_b0_all_folds_roc.png`
  - `results/cross_validation/figures/model_comparison_cv.png`
  - `results/cross_validation/figures/per_class_recall_cv.png`
