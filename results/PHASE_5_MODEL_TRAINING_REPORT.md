# COLONVISION AI — Phase 5 Model Training & Evaluation Report

**Date:** 2026-10-07 18:29:49  
**Pipeline:** Transfer Learning (ImageNet Pretrained) Differential Classification  
**Target Classes:** `colorectal_cancer` vs `colon_diverticula`  
**Device:** CPU  
**Random Seed:** 42  

---

## 1. Dataset & Split Configuration

| Split | Colorectal Cancer | Colon Diverticula | Total Images | Cancer : Diverticula Ratio |
| :--- | :---: | :---: | :---: | :---: |
| **Train** (70%) | 97 | 20 | **117** | 4.85 : 1 |
| **Validation** (15%) | 21 | 4 | **25** | 5.25 : 1 |
| **Test** (15%) | 21 | 5 | **26** | 4.20 : 1 |
| **Total** | **139** | **29** | **168** | 4.79 : 1 |

> **Class Imbalance Strategy:** Loss is weighted by inverse class frequency: `w_diverticula = 117 / (2 * 20) = 2.925`, `w_cancer = 117 / (2 * 97) = 0.603`. Augmentations (random horizontal flip, rotation +/-15 deg, color jitter, resized crop) applied to **TRAIN ONLY**.

---

## 2. Models Evaluated

1. **ResNet-50** (`resnet50`, ~23.5M params): Deep residual network with bottleneck blocks; head replaced with Dropout(0.4) + Linear(2048, 2); layer4 fine-tuned in Phase 2.
2. **DenseNet-121** (`densenet121`, ~6.96M params): Densely connected convolutional networks with feature reuse; classifier replaced with Dropout(0.4) + Linear(1024, 2); denseblock4 + norm5 fine-tuned in Phase 2.
3. **EfficientNet-B0** (`efficientnet_b0`, ~4.01M params): Efficient compound-scaled architecture with MBConv blocks; classifier replaced with Dropout(0.4) + Linear(1280, 2); top 3 feature stages fine-tuned in Phase 2.

---

## 3. Validation Set Performance (Model Selection)

Validation set (25 images: 21 cancer, 4 diverticula) used strictly for checkpoint selection and early stopping.

| Model | Best Epoch | Val Accuracy | Val Macro F1 | Val Weighted F1 | Val ROC-AUC | Diverticula Recall | Diverticula Precision | Diverticula F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **resnet50** | 4 | 1.0000 | **1.0000** | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 1.0000 |
| **densenet121** | 15 | 0.9600 | **0.9322** | 0.9617 | 0.9881 | **1.0000** | 0.8000 | 0.8889 |
| **efficientnet_b0** | 17 | 0.9600 | **0.9322** | 0.9617 | 0.9881 | **1.0000** | 0.8000 | 0.8889 |

---

## 4. Test Set Performance (Unseen Evaluation)

Test set (26 images: 21 cancer, 5 diverticula) evaluated once on the best checkpoint of each model.

| Model | Test Accuracy | Test Macro F1 | Test Weighted F1 | Test ROC-AUC | Diverticula Recall | Diverticula F1 | Cancer Recall | Cancer F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **resnet50** | 1.0000 | **1.0000** | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| **densenet121** | 0.9615 | **0.9328** | 0.9598 | 1.0000 | **0.8000** | 0.8889 | 1.0000 | 0.9767 |
| **efficientnet_b0** | 0.9615 | **0.9328** | 0.9598 | 0.9333 | **0.8000** | 0.8889 | 1.0000 | 0.9767 |

---

## 5. Selected Best Architecture: `resnet50`

- **Selection Criterion:** Selected based on highest validation Macro F1 score, validation colon diverticula recall, and ROC-AUC.
- **Model Weights Checkpoint:** `models/resnet50_best.pth`
- **Key Strength:** Robust differential features between colorectal malignancy and benign diverticular pouches.

---

## 6. Generated Visual Artifacts

- **Training Curves:** `results/training_curves/<model>_curves.png` (Loss & Accuracy over epochs)
- **Confusion Matrices:** `results/confusion_matrices/<model>_val_cm.png`, `<model>_test_cm.png`
- **ROC Curves:** `results/roc_curves/<model>_val_roc.png`, `<model>_test_roc.png`
- **Metrics Data:** `results/metrics/<model>_metrics.json`, `results/model_comparison.csv`

---

## 7. Medical & Methodological Notes

> [!IMPORTANT]
> 1. **Extreme Minority Sample Size:** The dataset contains only 29 colon diverticula images total (5 in test set). Each test misclassification of diverticula alters the minority recall by exactly 20 percentage points.
> 2. **Clinical Applicability:** While transfer-learning features effectively distinguish mucosal distortion (malignancy) from mucosal outpouching (diverticular orifice), these results represent a proof-of-concept benchmark and must undergo clinical multi-center validation before diagnostic deployment.
> 3. **Explainability Next Steps:** The next phase will apply Grad-CAM (Gradient-weighted Class Activation Mapping) on the selected best model to verify that classifications correspond to authentic endoscopic pathology rather than peripheral artifacts.
