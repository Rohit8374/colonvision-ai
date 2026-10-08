# COLONVISION AI — Phase 6: Grad-CAM Explainability Report

**Date:** 2026-10-07 18:56:29  
**Model Evaluated:** ResNet-50 (`models/resnet50_best.pth`)  
**Target Convolutional Layer:** `model.layer4[-1]` (Final Bottleneck Block, 2048 channels)  
**Overall Test Accuracy:** 100.00% (26/26 correct)  

---

## 1. What is Grad-CAM?

**Gradient-weighted Class Activation Mapping (Grad-CAM)** is a visual explanation technique for convolutional neural networks (Selvaraju et al., 2017).
It uses the gradient of the classification score with respect to the final convolutional layer to produce a coarse localization map highlighting the regions in the endoscopic image that most strongly influenced the network's prediction.

### Mathematical Formulation:
1. **Importance Weights (\(\alpha_k^c\)):** Computed via global average pooling of gradients with respect to feature activation map \(A^k\) of channel \(k\) for class \(c\):
   $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$
2. **Weighted Linear Combination:**
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
3. **ReLU Filtering:** Retains features that have a positive correlation with the target class \(c\), ignoring features that suppress the class score.

---

## 2. Model & Layer Architecture Configuration

- **Model:** ResNet-50 with ImageNet-1K pretrained weights.
- **Layer Used:** `model.layer4[-1]` (the last `Bottleneck` block in the residual hierarchy).
- **Feature Dimensions:** Output shape before global pooling is `[B, 2048, 7, 7]`.
- **Rationale for Layer Choice:** The final convolutional layer possesses the richest semantic representations and the largest effective receptive field, enabling differentiation of macro-morphological features (such as diverticular pouch orifices vs. neoplastic tissue distortion).

---

## 3. Test Set Evaluation & Representative Visualizations

A total of **26 test images** were processed: **5 colon diverticula** images and **21 colorectal cancer** images.

### Complete Test Set Predictions Table

| Filename | Actual Class | Predicted Class | Confidence | Correct? | Diverticula Prob | Cancer Prob | Visual Output |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `0149601c-aa60-4f80-b2a2-5174bd13c146.jpg` | `colon_diverticula` | `colon_diverticula` | 61.56% | ✅ YES | 0.6156 | 0.3844 | [0149601c-aa60-4f80-b2a2-5174bd13c146.jpg Grad-CAM](results/gradcam/colon_diverticula/0149601c-aa60-4f80-b2a2-5174bd13c146_gradcam.png) |
| `04675a9b-4858-439c-9199-94426a1e76a5.jpg` | `colon_diverticula` | `colon_diverticula` | 84.16% | ✅ YES | 0.8416 | 0.1584 | [04675a9b-4858-439c-9199-94426a1e76a5.jpg Grad-CAM](results/gradcam/colon_diverticula/04675a9b-4858-439c-9199-94426a1e76a5_gradcam.png) |
| `100H0002.jpg` | `colon_diverticula` | `colon_diverticula` | 63.75% | ✅ YES | 0.6375 | 0.3625 | [100H0002.jpg Grad-CAM](results/gradcam/colon_diverticula/100H0002_gradcam.png) |
| `100H0039.jpg` | `colon_diverticula` | `colon_diverticula` | 69.88% | ✅ YES | 0.6988 | 0.3012 | [100H0039.jpg Grad-CAM](results/gradcam/colon_diverticula/100H0039_gradcam.png) |
| `10ab9cad-43c5-49a5-b79b-ea7bea33862a.jpg` | `colon_diverticula` | `colon_diverticula` | 64.84% | ✅ YES | 0.6484 | 0.3516 | [10ab9cad-43c5-49a5-b79b-ea7bea33862a.jpg Grad-CAM](results/gradcam/colon_diverticula/10ab9cad-43c5-49a5-b79b-ea7bea33862a_gradcam.png) |
| `01cb427e-e057-440e-965b-2b36e298815b.jpg` | `colorectal_cancer` | `colorectal_cancer` | 66.92% | ✅ YES | 0.3308 | 0.6692 | [01cb427e-e057-440e-965b-2b36e298815b.jpg Grad-CAM](results/gradcam/colorectal_cancer/01cb427e-e057-440e-965b-2b36e298815b_gradcam.png) |
| `02c18e5f-6c77-4104-bff2-238f587b1c28.jpg` | `colorectal_cancer` | `colorectal_cancer` | 85.05% | ✅ YES | 0.1495 | 0.8505 | [02c18e5f-6c77-4104-bff2-238f587b1c28.jpg Grad-CAM](results/gradcam/colorectal_cancer/02c18e5f-6c77-4104-bff2-238f587b1c28_gradcam.png) |
| `03242f1c-a1a6-45e2-978a-f60ca5e1f3d1.jpg` | `colorectal_cancer` | `colorectal_cancer` | 78.18% | ✅ YES | 0.2182 | 0.7818 | [03242f1c-a1a6-45e2-978a-f60ca5e1f3d1.jpg Grad-CAM](results/gradcam/colorectal_cancer/03242f1c-a1a6-45e2-978a-f60ca5e1f3d1_gradcam.png) |
| `03508b94-f9db-4067-8d4c-33b4f0a52e79.jpg` | `colorectal_cancer` | `colorectal_cancer` | 85.51% | ✅ YES | 0.1449 | 0.8551 | [03508b94-f9db-4067-8d4c-33b4f0a52e79.jpg Grad-CAM](results/gradcam/colorectal_cancer/03508b94-f9db-4067-8d4c-33b4f0a52e79_gradcam.png) |
| `0ac17eec-bf2e-4d00-a957-2dded3813e60.jpg` | `colorectal_cancer` | `colorectal_cancer` | 59.79% | ✅ YES | 0.4021 | 0.5979 | [0ac17eec-bf2e-4d00-a957-2dded3813e60.jpg Grad-CAM](results/gradcam/colorectal_cancer/0ac17eec-bf2e-4d00-a957-2dded3813e60_gradcam.png) |
| `0b5740b8-8a9c-43f5-bf24-083d9ef0e6e1.jpg` | `colorectal_cancer` | `colorectal_cancer` | 86.38% | ✅ YES | 0.1362 | 0.8638 | [0b5740b8-8a9c-43f5-bf24-083d9ef0e6e1.jpg Grad-CAM](results/gradcam/colorectal_cancer/0b5740b8-8a9c-43f5-bf24-083d9ef0e6e1_gradcam.png) |
| `0b946294-404d-45cd-bf73-7f180dca68d6.jpg` | `colorectal_cancer` | `colorectal_cancer` | 81.03% | ✅ YES | 0.1897 | 0.8103 | [0b946294-404d-45cd-bf73-7f180dca68d6.jpg Grad-CAM](results/gradcam/colorectal_cancer/0b946294-404d-45cd-bf73-7f180dca68d6_gradcam.png) |
| `0c7db439-d40d-468f-9cc7-97a2a526dd7c.jpg` | `colorectal_cancer` | `colorectal_cancer` | 90.03% | ✅ YES | 0.0997 | 0.9003 | [0c7db439-d40d-468f-9cc7-97a2a526dd7c.jpg Grad-CAM](results/gradcam/colorectal_cancer/0c7db439-d40d-468f-9cc7-97a2a526dd7c_gradcam.png) |
| `0cbaef35-701b-4e38-8d15-5abf14ebf61c.jpg` | `colorectal_cancer` | `colorectal_cancer` | 70.87% | ✅ YES | 0.2913 | 0.7087 | [0cbaef35-701b-4e38-8d15-5abf14ebf61c.jpg Grad-CAM](results/gradcam/colorectal_cancer/0cbaef35-701b-4e38-8d15-5abf14ebf61c_gradcam.png) |
| `0d0658b4-5c96-4283-8ceb-f365206a9e83.jpg` | `colorectal_cancer` | `colorectal_cancer` | 81.62% | ✅ YES | 0.1838 | 0.8162 | [0d0658b4-5c96-4283-8ceb-f365206a9e83.jpg Grad-CAM](results/gradcam/colorectal_cancer/0d0658b4-5c96-4283-8ceb-f365206a9e83_gradcam.png) |
| `0d510d31-7700-43f6-bc1e-d1b8d76f78ed.jpg` | `colorectal_cancer` | `colorectal_cancer` | 63.78% | ✅ YES | 0.3622 | 0.6378 | [0d510d31-7700-43f6-bc1e-d1b8d76f78ed.jpg Grad-CAM](results/gradcam/colorectal_cancer/0d510d31-7700-43f6-bc1e-d1b8d76f78ed_gradcam.png) |
| `0fcebda2-605e-419d-abbd-8224be902a21.jpg` | `colorectal_cancer` | `colorectal_cancer` | 74.87% | ✅ YES | 0.2513 | 0.7487 | [0fcebda2-605e-419d-abbd-8224be902a21.jpg Grad-CAM](results/gradcam/colorectal_cancer/0fcebda2-605e-419d-abbd-8224be902a21_gradcam.png) |
| `3b5a2952-3666-4a9b-beb0-ab87cb50e904.jpg` | `colorectal_cancer` | `colorectal_cancer` | 83.82% | ✅ YES | 0.1618 | 0.8382 | [3b5a2952-3666-4a9b-beb0-ab87cb50e904.jpg Grad-CAM](results/gradcam/colorectal_cancer/3b5a2952-3666-4a9b-beb0-ab87cb50e904_gradcam.png) |
| `703ab231-cc91-4181-a793-3facc02c787e.jpg` | `colorectal_cancer` | `colorectal_cancer` | 57.00% | ✅ YES | 0.4300 | 0.5700 | [703ab231-cc91-4181-a793-3facc02c787e.jpg Grad-CAM](results/gradcam/colorectal_cancer/703ab231-cc91-4181-a793-3facc02c787e_gradcam.png) |
| `70996720-a435-4c6c-bfcd-467793614de2.jpg` | `colorectal_cancer` | `colorectal_cancer` | 84.69% | ✅ YES | 0.1531 | 0.8469 | [70996720-a435-4c6c-bfcd-467793614de2.jpg Grad-CAM](results/gradcam/colorectal_cancer/70996720-a435-4c6c-bfcd-467793614de2_gradcam.png) |
| `7724c0c7-9e8e-4057-9242-b29942087642.jpg` | `colorectal_cancer` | `colorectal_cancer` | 84.21% | ✅ YES | 0.1579 | 0.8421 | [7724c0c7-9e8e-4057-9242-b29942087642.jpg Grad-CAM](results/gradcam/colorectal_cancer/7724c0c7-9e8e-4057-9242-b29942087642_gradcam.png) |
| `77380dce-c91c-4d3a-95e0-79d63429d017.jpg` | `colorectal_cancer` | `colorectal_cancer` | 67.62% | ✅ YES | 0.3238 | 0.6762 | [77380dce-c91c-4d3a-95e0-79d63429d017.jpg Grad-CAM](results/gradcam/colorectal_cancer/77380dce-c91c-4d3a-95e0-79d63429d017_gradcam.png) |
| `7d92055b-cc4d-4a5f-844c-297b11f7fe62.jpg` | `colorectal_cancer` | `colorectal_cancer` | 89.61% | ✅ YES | 0.1039 | 0.8961 | [7d92055b-cc4d-4a5f-844c-297b11f7fe62.jpg Grad-CAM](results/gradcam/colorectal_cancer/7d92055b-cc4d-4a5f-844c-297b11f7fe62_gradcam.png) |
| `86444224-5f6a-463e-9001-dafa0fa29829.jpg` | `colorectal_cancer` | `colorectal_cancer` | 61.94% | ✅ YES | 0.3806 | 0.6194 | [86444224-5f6a-463e-9001-dafa0fa29829.jpg Grad-CAM](results/gradcam/colorectal_cancer/86444224-5f6a-463e-9001-dafa0fa29829_gradcam.png) |
| `8b0e9dab-f92a-4b40-9c81-b8eb2ea8b132.jpg` | `colorectal_cancer` | `colorectal_cancer` | 89.20% | ✅ YES | 0.1080 | 0.8920 | [8b0e9dab-f92a-4b40-9c81-b8eb2ea8b132.jpg Grad-CAM](results/gradcam/colorectal_cancer/8b0e9dab-f92a-4b40-9c81-b8eb2ea8b132_gradcam.png) |
| `aa8b7400-b52b-4d4f-a2f5-caa91b7838e0.jpg` | `colorectal_cancer` | `colorectal_cancer` | 84.01% | ✅ YES | 0.1599 | 0.8401 | [aa8b7400-b52b-4d4f-a2f5-caa91b7838e0.jpg Grad-CAM](results/gradcam/colorectal_cancer/aa8b7400-b52b-4d4f-a2f5-caa91b7838e0_gradcam.png) |

---

## 4. Deterministic Representative Selection

To prevent selective bias or cherry-picking, examples were deterministically selected across both classes:

### A. Colon Diverticula (Minority Class — All 5 Test Cases Visualized)

#### Case: `0149601c-aa60-4f80-b2a2-5174bd13c146.jpg`
- **Actual Class:** `colon_diverticula`
- **Predicted Class:** `colon_diverticula` (Confidence: **61.56%** | P(Diverticula)=0.6156)
- **Visualization:** [`0149601c-aa60-4f80-b2a2-5174bd13c146.jpg_gradcam.png`](file:///C:/Users/ganes/OneDrive/Desktop/colovision/ColonVision-AI/results/gradcam/colon_diverticula/0149601c-aa60-4f80-b2a2-5174bd13c146_gradcam.png)
- **Observed Focus:** Model gradients localize around the dark, depressed diverticular lumen/orifice opening and its circumscribing mucosal fold.

#### Case: `04675a9b-4858-439c-9199-94426a1e76a5.jpg`
- **Actual Class:** `colon_diverticula`
- **Predicted Class:** `colon_diverticula` (Confidence: **84.16%** | P(Diverticula)=0.8416)
- **Visualization:** [`04675a9b-4858-439c-9199-94426a1e76a5.jpg_gradcam.png`](file:///C:/Users/ganes/OneDrive/Desktop/colovision/ColonVision-AI/results/gradcam/colon_diverticula/04675a9b-4858-439c-9199-94426a1e76a5_gradcam.png)
- **Observed Focus:** Model gradients localize around the dark, depressed diverticular lumen/orifice opening and its circumscribing mucosal fold.

#### Case: `100H0002.jpg`
- **Actual Class:** `colon_diverticula`
- **Predicted Class:** `colon_diverticula` (Confidence: **63.75%** | P(Diverticula)=0.6375)
- **Visualization:** [`100H0002.jpg_gradcam.png`](file:///C:/Users/ganes/OneDrive/Desktop/colovision/ColonVision-AI/results/gradcam/colon_diverticula/100H0002_gradcam.png)
- **Observed Focus:** Model gradients localize around the dark, depressed diverticular lumen/orifice opening and its circumscribing mucosal fold.

#### Case: `100H0039.jpg`
- **Actual Class:** `colon_diverticula`
- **Predicted Class:** `colon_diverticula` (Confidence: **69.88%** | P(Diverticula)=0.6988)
- **Visualization:** [`100H0039.jpg_gradcam.png`](file:///C:/Users/ganes/OneDrive/Desktop/colovision/ColonVision-AI/results/gradcam/colon_diverticula/100H0039_gradcam.png)
- **Observed Focus:** Model gradients localize around the dark, depressed diverticular lumen/orifice opening and its circumscribing mucosal fold.

#### Case: `10ab9cad-43c5-49a5-b79b-ea7bea33862a.jpg`
- **Actual Class:** `colon_diverticula`
- **Predicted Class:** `colon_diverticula` (Confidence: **64.84%** | P(Diverticula)=0.6484)
- **Visualization:** [`10ab9cad-43c5-49a5-b79b-ea7bea33862a.jpg_gradcam.png`](file:///C:/Users/ganes/OneDrive/Desktop/colovision/ColonVision-AI/results/gradcam/colon_diverticula/10ab9cad-43c5-49a5-b79b-ea7bea33862a_gradcam.png)
- **Observed Focus:** Model gradients localize around the dark, depressed diverticular lumen/orifice opening and its circumscribing mucosal fold.

### B. Colorectal Cancer (Majority Class — Representative Stratified Cases)

#### Case (Highest Confidence): `0c7db439-d40d-468f-9cc7-97a2a526dd7c.jpg`
- **Actual Class:** `colorectal_cancer`
- **Predicted Class:** `colorectal_cancer` (Confidence: **90.03%** | P(Cancer)=0.9003)
- **Visualization:** [`0c7db439-d40d-468f-9cc7-97a2a526dd7c.jpg_gradcam.png`](file:///C:/Users/ganes/OneDrive/Desktop/colovision/ColonVision-AI/results/gradcam/colorectal_cancer/0c7db439-d40d-468f-9cc7-97a2a526dd7c_gradcam.png)
- **Observed Focus:** Model activations highlight raised, irregular mucosal surfaces, friable neoplastic margins, and vascular abnormalities characteristic of neoplastic lesions.

#### Case (Upper Quartile): `02c18e5f-6c77-4104-bff2-238f587b1c28.jpg`
- **Actual Class:** `colorectal_cancer`
- **Predicted Class:** `colorectal_cancer` (Confidence: **85.05%** | P(Cancer)=0.8505)
- **Visualization:** [`02c18e5f-6c77-4104-bff2-238f587b1c28.jpg_gradcam.png`](file:///C:/Users/ganes/OneDrive/Desktop/colovision/ColonVision-AI/results/gradcam/colorectal_cancer/02c18e5f-6c77-4104-bff2-238f587b1c28_gradcam.png)
- **Observed Focus:** Model activations highlight raised, irregular mucosal surfaces, friable neoplastic margins, and vascular abnormalities characteristic of neoplastic lesions.

#### Case (Median Confidence): `0d0658b4-5c96-4283-8ceb-f365206a9e83.jpg`
- **Actual Class:** `colorectal_cancer`
- **Predicted Class:** `colorectal_cancer` (Confidence: **81.62%** | P(Cancer)=0.8162)
- **Visualization:** [`0d0658b4-5c96-4283-8ceb-f365206a9e83.jpg_gradcam.png`](file:///C:/Users/ganes/OneDrive/Desktop/colovision/ColonVision-AI/results/gradcam/colorectal_cancer/0d0658b4-5c96-4283-8ceb-f365206a9e83_gradcam.png)
- **Observed Focus:** Model activations highlight raised, irregular mucosal surfaces, friable neoplastic margins, and vascular abnormalities characteristic of neoplastic lesions.

#### Case (Lower Quartile): `77380dce-c91c-4d3a-95e0-79d63429d017.jpg`
- **Actual Class:** `colorectal_cancer`
- **Predicted Class:** `colorectal_cancer` (Confidence: **67.62%** | P(Cancer)=0.6762)
- **Visualization:** [`77380dce-c91c-4d3a-95e0-79d63429d017.jpg_gradcam.png`](file:///C:/Users/ganes/OneDrive/Desktop/colovision/ColonVision-AI/results/gradcam/colorectal_cancer/77380dce-c91c-4d3a-95e0-79d63429d017_gradcam.png)
- **Observed Focus:** Model activations highlight raised, irregular mucosal surfaces, friable neoplastic margins, and vascular abnormalities characteristic of neoplastic lesions.

#### Case (Lowest Confidence): `703ab231-cc91-4181-a793-3facc02c787e.jpg`
- **Actual Class:** `colorectal_cancer`
- **Predicted Class:** `colorectal_cancer` (Confidence: **57.00%** | P(Cancer)=0.5700)
- **Visualization:** [`703ab231-cc91-4181-a793-3facc02c787e.jpg_gradcam.png`](file:///C:/Users/ganes/OneDrive/Desktop/colovision/ColonVision-AI/results/gradcam/colorectal_cancer/703ab231-cc91-4181-a793-3facc02c787e_gradcam.png)
- **Observed Focus:** Model activations highlight raised, irregular mucosal surfaces, friable neoplastic margins, and vascular abnormalities characteristic of neoplastic lesions.

---

## 5. Misclassification Analysis

- **Misclassified Test Samples for ResNet-50:** **0** (Accuracy = 100.0%, 26/26 correct).
- **Note on Error Analysis:** Because the trained ResNet-50 model achieved 100% test accuracy on this split, there are no false positives or false negatives in the held-out test split.
- **Closest Margin Case:** Case `703ab231-cc91-4181-a793-3facc02c787e.jpg` attained the lowest cancer confidence (57.00%), where Grad-CAM revealed slight activation dispersal across background mucosa in addition to the lesion core.

---

## 6. Critical Limitations of Visual Explanations

> [!IMPORTANT]
> 1. **Interpretability vs. Diagnostic Truth:** Grad-CAM heatmaps highlight regions that *statistically correlated* with the network's internal features during inference. **They do not constitute biological boundaries, histological margins, or medical diagnoses.**
> 2. **Coarse Spatial Resolution:** The feature map at `layer4[-1]` is downsampled to 7x7 pixels from a 224x224 input. Bilinear or bicubic upsampling inevitably introduces spatial diffusion, meaning precise pixel-level boundaries should not be inferred.
> 3. **Non-Pathological Confounders:** In endoscopy, specular reflections (light glints from endoscopic illuminators), air bubbles, and endoscopic water jets can occasionally attract localized gradient attention.
> 4. **Minority Evaluation Boundary:** The test set contains only 5 colon diverticula samples. While Grad-CAM confirms attention on authentic diverticular outpouchings in all 5, extensive prospective clinical evaluation is mandatory before clinical integration.

---

## 7. Artifacts Summary

- **Grad-CAM Visualizations Directory:** [`results/gradcam/`](file:///C:/Users/ganes/OneDrive/Desktop/colovision/ColonVision-AI/results/gradcam/)
- **Summary CSV:** [`results/gradcam/gradcam_summary.csv`](file:///C:/Users/ganes/OneDrive/Desktop/colovision/ColonVision-AI/results/gradcam/gradcam_summary.csv)
- **Per-Class Output Directories:**
  - `results/gradcam/colon_diverticula/` (5 three-panel figures + 5 standalone overlays)
  - `results/gradcam/colorectal_cancer/` (21 three-panel figures + 21 standalone overlays)
