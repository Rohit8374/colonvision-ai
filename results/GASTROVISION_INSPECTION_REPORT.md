# COLONVISION AI — GASTROVISION DATASET INSPECTION REPORT

**Date**: October 7, 2026  
**Dataset Evaluated**: GastroVision Dataset  
**Authoritative Source**: Folder hierarchy at `C:\Users\ganes\Downloads\GastroVision\Gastrovision`  

---

## 1. Executive Summary & Core Metrics

- **Total Image Files**: `8,000` valid `.jpg` images (plus 3 non-image macOS `.DS_Store` hidden files)
- **Total Classes**: `27` distinct disease / anatomical finding categories
- **Target Classes Available**:
  - **Colorectal Cancer**: `139` images
  - **Colon Diverticula**: `29` images
- **Image Formats**: 100% JPEG (`8,000` images)
- **Resolution Range**:
  - Min: `720 x 576`
  - Max: `1920 x 1080`
  - Mean: `874 x 660`
  - Top common resolution: `768 x 576` (`3,890` images, 48.6%)
- **Corrupted / Unreadable Images**: `0` valid image corruptions (3 macOS `.DS_Store` system files skipped)
- **Duplicate Images (MD5 Hash)**: `1` duplicate pair (`ba615bcd-2f99-4a12-884d-d9cd2c04c2a1.jpg` and `ckda1fpc5000l3a5s17a45xql.jpg` in `Colon polyps`)

---

## 2. Complete Class Distribution

| # | Class Name (Folder) | Image Count | Percentage | Target Class Status |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `Normal mucosa and vascular pattern in the large bowel` | `1,467` | 18.34% | Baseline Normal mucosa |
| 2 | `Accessory tools` | `1,266` | 15.83% | Artifact / Instrument |
| 3 | `Normal stomach` | `969` | 12.11% | Upper GI Normal |
| 4 | `Small bowel_terminal ileum` | `846` | 10.58% | Lower GI Normal |
| 5 | `Colon polyps` | `820` | 10.25% | Pre-cancerous pathology |
| 6 | `Pylorus` | `393` | 4.91% | Upper GI Landmark |
| 7 | `Gastroesophageal_junction_normal z-line` | `330` | 4.13% | Upper GI Landmark |
| 8 | `Dyed-resection-margins` | `246` | 3.08% | Intervention |
| 9 | `Duodenal bulb` | `205` | 2.56% | Upper GI Landmark |
| 10 | `Ileocecal valve` | `200` | 2.50% | Lower GI Landmark |
| 11 | `Blood in lumen` | `171` | 2.14% | Finding |
| 12 | `Normal esophagus` | `140` | 1.75% | Upper GI Normal |
| 13 | **`Colorectal cancer`** | **`139`** | **1.74%** | **Target Class A (Cancer)** |
| 14 | `Dyed-lifted-polyps` | `141` | 1.76% | Intervention |
| 15 | `Cecum` | `113` | 1.41% | Lower GI Landmark |
| 16 | `Esophagitis` | `107` | 1.34% | Upper GI Pathology |
| 17 | `Barrett's esophagus` | `95` | 1.19% | Upper GI Pathology |
| 18 | `Resected polyps` | `93` | 1.16% | Intervention |
| 19 | `Retroflex rectum` | `67` | 0.84% | Lower GI Landmark |
| 20 | `Gastric polyps` | `66` | 0.83% | Upper GI Pathology |
| 21 | **`Colon diverticula`** | **`29`** | **0.36%** | **Target Class B (Diverticula)** |
| 22 | `Mucosal inflammation large bowel` | `29` | 0.36% | Lower GI Inflammation |
| 23 | `Resection margins` | `26` | 0.33% | Intervention |
| 24 | `Angiectasia` | `17` | 0.21% | Vascular pathology |
| 25 | `Erythema` | `15` | 0.19% | Inflammatory finding |
| 26 | `Esophageal varices` | `7` | 0.09% | Upper GI Vascular |
| 27 | `Ulcer` | `6` | 0.08% | Mucosal lesion |

---

## 3. Structural & Metadata Evaluation

- **Metadata Files**: No standalone CSV or JSON metadata file exists inside `Gastrovision.zip`. Ground truth labels are entirely established by folder hierarchy names.
- **Official Train/Val/Test Splits**: No official split manifests or subdirectories exist in the archive.
- **Patient / Source Identifiers**: Filenames use randomized UUID strings (`ba615bcd-2f99-4a12-884d-d9cd2c04c2a1.jpg`). No explicit patient or video session IDs are present in filenames.
- **Folder Structure**: Single top-level directory (`Gastrovision/`) containing 27 subdirectories representing each class.

---

## 4. Key Findings for ColonVision AI

1. **Direct Label Availability**: Unlike HyperKvasir, GastroVision contains **explicitly labeled samples** for both of our target conditions:
   - `Colorectal cancer`: **139 images**
   - `Colon diverticula`: **29 images**
2. **Sample Count Limitation**: `29` images for `Colon diverticula` is a small sample size for deep learning from scratch, but suitable for transfer learning, data augmentation, or cross-dataset merging with complementary open sources.
3. **Class Imbalance**: Imbalance ratio of `50.6 : 1` between majority class (`Normal mucosa...`, 1467 images) and `Colon diverticula` (29 images).

---

## 5. Compliance & Rule Checklist

- [x] No disease labels were visually guessed or inferred.
- [x] Class names were preserved exactly as defined by the dataset folders.
- [x] Polyps (`820` images) were NOT converted or renamed to colorectal cancer.
- [x] Impacted stool / other findings were NOT converted or renamed to colon diverticula.
- [x] No images were deleted, modified, duplicated, or augmented.
- [x] No data preprocessing was initiated.
- [x] No model training was initiated.
