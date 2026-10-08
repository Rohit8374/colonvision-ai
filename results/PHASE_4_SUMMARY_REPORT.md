# COLONVISION AI — PHASE 4 DATASET INSPECTION REPORT

**Date**: October 7, 2026  
**Dataset Evaluated**: HyperKvasir Labeled Images Dataset  
**Authoritative Source**: `C:\Users\ganes\Downloads\hyper-kvasir-labeled-images\labeled-images\image-labels.csv`  

---

## 1. Dataset Overview & Size
- **Total Images in Metadata CSV**: `10,662`
- **Total Valid Image Files on Disk**: `10,662` (100% match)
- **Corrupted / Unreadable Images**: `0`
- **Exact Duplicate Files (MD5 Hash)**: `0`
- **Image Formats**: 100% JPEG (`10,662` images)
- **Image Resolution Range**:
  - Min: `332 x 352`
  - Max: `1920 x 1079`
  - Mean: `817 x 685`
  - Top common resolution: `633 x 532` (`1,311` images)

---

## 2. Complete Class List & Organ Breakdown

The dataset contains **23 original classes** across **2 anatomical organs** (Lower GI and Upper GI) and **4 classification categories**:

| Organ | Images | Percentage |
| :--- | :--- | :--- |
| **Lower GI Tract** | `7,210` | 67.62% |
| **Upper GI Tract** | `3,452` | 32.38% |
| **Total** | `10,662` | 100.00% |

### Complete Class Distribution (`image-labels.csv`):

| # | Original Label | Organ | Classification Category | Image Count | Percentage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `bbps-2-3` | Lower GI | quality-of-mucosal-views | `1,148` | 10.77% |
| 2 | `polyps` | Lower GI | pathological-findings | `1,028` | 9.64% |
| 3 | `cecum` | Lower GI | anatomical-landmarks | `1,009` | 9.46% |
| 4 | `dyed-lifted-polyps` | Lower GI | therapeutic-interventions | `1,002` | 9.40% |
| 5 | `pylorus` | Upper GI | anatomical-landmarks | `999` | 9.37% |
| 6 | `dyed-resection-margins` | Lower GI | therapeutic-interventions | `989` | 9.28% |
| 7 | `z-line` | Upper GI | anatomical-landmarks | `932` | 8.74% |
| 8 | `retroflex-stomach` | Upper GI | anatomical-landmarks | `764` | 7.17% |
| 9 | `bbps-0-1` | Lower GI | quality-of-mucosal-views | `646` | 6.06% |
| 10 | `ulcerative-colitis-grade-2` | Lower GI | pathological-findings | `443` | 4.15% |
| 11 | `esophagitis-a` | Upper GI | pathological-findings | `403` | 3.78% |
| 12 | `retroflex-rectum` | Lower GI | anatomical-landmarks | `391` | 3.67% |
| 13 | `esophagitis-b-d` | Upper GI | pathological-findings | `260` | 2.44% |
| 14 | `ulcerative-colitis-grade-1` | Lower GI | pathological-findings | `201` | 1.89% |
| 15 | `ulcerative-colitis-grade-3` | Lower GI | pathological-findings | `133` | 1.25% |
| 16 | `impacted-stool` | Lower GI | quality-of-mucosal-views | `131` | 1.23% |
| 17 | `barretts-short-segment` | Upper GI | pathological-findings | `53` | 0.50% |
| 18 | `barretts` | Upper GI | pathological-findings | `41` | 0.38% |
| 19 | `ulcerative-colitis-grade-0-1` | Lower GI | pathological-findings | `35` | 0.33% |
| 20 | `ulcerative-colitis-grade-2-3` | Lower GI | pathological-findings | `28` | 0.26% |
| 21 | `ulcerative-colitis-grade-1-2` | Lower GI | pathological-findings | `11` | 0.10% |
| 22 | `ileum` | Lower GI | anatomical-landmarks | `9` | 0.08% |
| 23 | `hemorrhoids` | Lower GI | pathological-findings | `6` | 0.06% |

---

## 3. Target Class Verification

### A. Colorectal Cancer
- **Directly Labeled Classes Found**: **`0`** (`NONE`)
- **Direct Sample Count**: `0`
- **Assessment**: HyperKvasir does **NOT** contain any class labeled as colorectal cancer, carcinoma, or adenocarcinoma. 
- *Rule Compliance*: `polyps` (`1,028` images) and `dyed-lifted-polyps` (`1,002` images) are labeled as polyps/therapeutic interventions and were **NOT** reclassified or assumed to be colorectal cancer.

### B. Diverticular Disease / Diverticula
- **Directly Labeled Classes Found**: **`0`** (`NONE`)
- **Direct Sample Count**: `0`
- **Assessment**: HyperKvasir does **NOT** contain any class labeled as diverticular disease, diverticulosis, or diverticulitis.
- *Rule Compliance*: Other lower GI findings (`impacted-stool`, `hemorrhoids`, `ulcerative-colitis`, `cecum`) were **NOT** treated as diverticular disease.

---

## 4. Class Imbalance Analysis
- **Imbalance Ratio**: `191.33 : 1` (Majority class `bbps-2-3` with `1,148` samples vs. minority class `hemorrhoids` with `6` samples).

---

## 5. Dataset Sufficiency & Limitations

### Is HyperKvasir Alone Sufficient for ColonVision AI?
> ❌ **NO**. HyperKvasir alone is **INSUFFICIENT** to train or evaluate the proposed binary differential classification of **Colorectal Cancer vs. Diverticular Disease**.

### Summary of Discovered Limitations:
1. **Absence of Primary Target Classes**: Neither Colorectal Cancer nor Diverticular Disease exists as a distinct, labeled class in HyperKvasir.
2. **Scope Overlap**: 32.38% (`3,452` images) of HyperKvasir consists of Upper GI images (stomach, esophagus), whereas colonoscopy targets Lower GI.
3. **High Class Imbalance**: Imbalance ratio of `191.33:1` across existing non-target classes.

---

## 6. Recommended Next Steps

1. **Dataset Acquisition Options**:
   - **Option A (Multi-Dataset Integration)**: Acquire secondary public medical datasets containing confirmed images of colorectal carcinoma/cancer and diverticulosis/diverticulitis (e.g., specific clinical open access repositories or target GI datasets).
   - **Option B (Scope Adjustment)**: Under academic supervisor guidance, refine the project objective to leverage existing HyperKvasir pathological classes (e.g., differential classification of **Polyps vs. Ulcerative Colitis** or **Pathological Findings vs. Normal Mucosa** with Grad-CAM explainability).
2. **Await Project Approval**: Stop and wait for user approval of dataset strategy before starting any data preprocessing or model training.
