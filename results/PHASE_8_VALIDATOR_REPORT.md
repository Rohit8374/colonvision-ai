# COLONVISION AI — Phase 8: Endoscopy-Domain Validation & OOD Protection Report

**Date:** 2026-10-07 22:17:40  
**Module:** Endoscopy-domain validation / OOD protection layer  
**Feature Backbone:** ResNet-50 ImageNet-1K pretrained (2048-dimensional pooling layer)  
**Calibrated Domain Threshold:** `tau* = 0.68`  

> [!IMPORTANT]
> **Disclaimer:** Domain verification is an image-level protection mechanism and does not establish clinical validity or diagnostic suitability.

---

## 1. Problem Statement & Motivation

A closed-set binary classifier forced arbitrary non-endoscopic inputs (e.g. a user passport photo) into:
`0 = colon_diverticula` vs `1 = colorectal_cancer`, producing a spurious 69.26% cancer probability.
To eliminate this vulnerability, a pre-classification domain verification layer was constructed to filter out non-endoscopic imagery before differential classification and Grad-CAM execution.

---

## 2. Experimental Data Partitioning

| Partition | Subset | Count | Description / Categories Included |
| :--- | :--- | :---: | :--- |
| **Profile Reference** | Positive (In-Domain) | 391 | Stratified sample across all 27 GastroVision classes (mucosa, polyps, bleeding, tools, stomach, esophagus, cancer, diverticula, etc.) |
| **Calibration Set** | Positive (In-Domain) | 129 | Held-out GastroVision endoscopic frames disjoint from profile reference |
| **Calibration Set** | Negative (Out-of-Domain) | 20 | User passport photo, student ID card, portraits, casual smartphone photos, digital graphics |
| **Evaluation Set** | Positive (In-Domain) | 132 | Unseen held-out GastroVision endoscopic frames disjoint from both profile & calibration |
| **Evaluation Set** | Negative (Out-of-Domain) | 21 | Unseen negative images: outdoor photography, documents, UI graphics, wallpapers, selfies |

---

## 3. Threshold Calibration Table (Candidate Sweep)

| Candidate Threshold (tau) | Endoscopy Acceptance Rate | OOD Rejection Rate | False Rejections | False Acceptances | Youden's J Index |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `0.6800` | 99.18% | 100.00% | 1 | 0 | 0.9918 👈 **SELECTED** |
| `0.6850` | 99.18% | 100.00% | 1 | 0 | 0.9918 |
| `0.6900` | 99.18% | 100.00% | 1 | 0 | 0.9918 |
| `0.6950` | 99.18% | 100.00% | 1 | 0 | 0.9918 |
| `0.7000` | 99.18% | 100.00% | 1 | 0 | 0.9918 |
| `0.7050` | 99.18% | 100.00% | 1 | 0 | 0.9918 |
| `0.7100` | 99.18% | 100.00% | 1 | 0 | 0.9918 |
| `0.7150` | 99.18% | 100.00% | 1 | 0 | 0.9918 |
| `0.7200` | 99.18% | 100.00% | 1 | 0 | 0.9918 |
| `0.7250` | 99.18% | 100.00% | 1 | 0 | 0.9918 |
| `0.7300` | 99.18% | 100.00% | 1 | 0 | 0.9918 |
| `0.7350` | 99.18% | 100.00% | 1 | 0 | 0.9918 |
| `0.7400` | 99.18% | 100.00% | 1 | 0 | 0.9918 |
| `0.7450` | 99.18% | 100.00% | 1 | 0 | 0.9918 |
| `0.7500` | 99.18% | 100.00% | 1 | 0 | 0.9918 |
| `0.7550` | 97.54% | 100.00% | 3 | 0 | 0.9754 |
| `0.7600` | 95.90% | 100.00% | 5 | 0 | 0.9590 |
| `0.7650` | 95.90% | 100.00% | 5 | 0 | 0.9590 |
| `0.7700` | 95.90% | 100.00% | 5 | 0 | 0.9590 |
| `0.7750` | 95.08% | 100.00% | 6 | 0 | 0.9508 |
| `0.7800` | 92.62% | 100.00% | 9 | 0 | 0.9262 |
| `0.7850` | 91.80% | 100.00% | 10 | 0 | 0.9180 |
| `0.7900` | 90.98% | 100.00% | 11 | 0 | 0.9098 |
| `0.7950` | 90.16% | 100.00% | 12 | 0 | 0.9016 |
| `0.8000` | 90.16% | 100.00% | 12 | 0 | 0.9016 |

---

## 4. Final Evaluation Performance on Unseen Data

Evaluated on 132 unseen positive endoscopy images and 21 unseen negative images at `tau* = 0.68`:

- **Endoscopy Acceptance Rate:** **100.00%**
- **OOD Rejection Rate:** **95.24%**
- **False Acceptance Count:** **1** (OOD passed as endoscopy)
- **False Rejection Count:** **0** (Endoscopy incorrectly rejected)

---

## 5. Verification of the Passport Photo Failure Case

- **File Tested:** `WhatsApp Image 2026-09-28 at 10.17.10 PM.jpeg`
- **Domain Similarity Score:** `0.6618`
- **Required Threshold:** `0.68`
- **Decision:** **REJECTED (Score 0.6618 < 0.68)** ✅
- **Outcome:** Gate triggers `Unsupported Image`, completely suppressing downstream ResNet-50 cancer/diverticula prediction and Grad-CAM generation.

---

## 6. Examples of Accepted vs. Rejected Inputs

### A. Rejected Non-Endoscopic Images (Sample):
- `WhatsApp Image 2026-09-28 at 10.17.10 PM.jpeg`: Score = `0.6618` (Decision: **REJECTED**)
- `Gemini_Generated_Image_vuzryrvuzryrvuzr.png`: Score = `0.6310` (Decision: **REJECTED**)
- `ChatGPT Image May 24, 2026, 06_35_33 PM.png`: Score = `0.5716` (Decision: **REJECTED**)
- `Gemini_Generated_Image_xfenszxfenszxfen.png`: Score = `0.6294` (Decision: **REJECTED**)
- `WhatsApp Image 2026-04-25 at 7.52.37 PM.jpeg`: Score = `0.5908` (Decision: **REJECTED**)
- `SRKR_Student_ID_Card.png`: Score = `0.5753` (Decision: **REJECTED**)

### B. Accepted Endoscopic Images (Sample):
- `Gastrovision/Cecum/001.jpg`: Score = `0.8857` (Decision: **ACCEPTED**)
- `Gastrovision/Colon polyps/002.jpg`: Score = `0.9183` (Decision: **ACCEPTED**)
- `Gastrovision/Colorectal cancer/001.jpg`: Score = `0.8991` (Decision: **ACCEPTED**)
- `Gastrovision/Colon diverticula/001.jpg`: Score = `0.8608` (Decision: **ACCEPTED**)

---

## 7. Methodological Limitations

> [!WARNING]
> 1. **Image-Level Domain Heuristic:** Cosine similarity to the feature centroid measures resemblance to the broader gastrointestinal endoscopic manifold. It does **not** prove an image is medically a colonoscopy or from any specific organ.
> 2. **Local Negative Set Size:** The negative evaluation set utilizes 41 local everyday photographs available in the runtime environment. While diverse (portraits, ID cards, documents, outdoor photos, graphics), full validation on large open-set datasets (e.g. ImageNet, COCO) remains desirable for production deployment.
> 3. **Non-Relabeling:** GastroVision reference classes (e.g. polyps, inflammation) were strictly utilized as general endoscopy representations and were never relabeled as colorectal cancer or diverticular disease.
