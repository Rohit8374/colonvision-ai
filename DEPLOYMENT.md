# COLONVISION AI — Streamlit Community Cloud Deployment Guide

This guide describes how to deploy the **ColonVision AI** clinical research workspace to **Streamlit Community Cloud** via GitHub.

---

## 1. Prerequisites

- A [GitHub account](https://github.com)
- A [Streamlit Community Cloud account](https://share.streamlit.io) (signed in with GitHub)
- Git installed on your local computer

---

## 2. Step-by-Step Deployment Instructions

### Step 1: Create a New GitHub Repository
1. Go to [GitHub: New Repository](https://github.com/new).
2. Name your repository (e.g., `colonvision-ai` or `ColonVision-AI`).
3. Set visibility to **Public** (required for free Streamlit Community Cloud hosting) or **Private** (if your Streamlit Cloud plan supports private repos).
4. Do **not** initialize with a README, `.gitignore`, or license (the local project already contains these).
5. Click **Create repository**.

### Step 2: Initialize Git and Push Local Code
Open PowerShell or your terminal in the project directory:

```powershell
cd C:\Users\ganes\OneDrive\Desktop\colovision\ColonVision-AI

# 1. Initialize git repository (if not already initialized)
git init

# 2. Stage verified deployment files (.gitignore protects datasets & environments)
git add .

# 3. Commit files
git commit -m "feat: prepare ColonVision AI for Streamlit Cloud deployment"

# 4. Set branch name to main
git branch -M main

# 5. Link your GitHub remote repository (replace YOUR_USERNAME and YOUR_REPO)
git remote add origin https://github.com/YOUR_USERNAME/colonvision-ai.git

# 6. Push to GitHub
git push -u origin main
```

*(Note: GitHub will show an informational advisory for `resnet50_best.pth` because it is ~90 MB, but the push will succeed as it is under the 100 MB hard limit.)*

---

### Step 3: Deploy on Streamlit Community Cloud
1. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
2. Click **Create app** (or **New app**).
3. Choose **"Deploy a public app from GitHub"**.
4. Configure the deployment settings:
   - **Repository:** `YOUR_USERNAME/colonvision-ai`
   - **Branch:** `main`
   - **Main file path:** `app/app.py`
   - **App URL:** Customize if desired (e.g., `colonvision-ai.streamlit.app`)
5. Click **Advanced settings...** (Optional):
   - **Python version:** Select **`3.11`** (recommended to match local development).
6. Click **Deploy!**

---

### Step 4: Streamlit Secrets Configuration (Optional)
This application runs **100% self-contained** using local PyTorch weights and pre-calibrated OOD profile data.
- **No API keys or cloud database secrets are required** for normal operation.
- If future integrations (such as cloud storage or third-party APIs) are added, configure them securely in:
  - Streamlit Cloud App Settings $\rightarrow$ **Secrets** $\rightarrow$ `.streamlit/secrets.toml`.

---

### Step 5: Test the Public Application
Once the build completes (takes ~2–3 minutes for initial container setup):
1. **Verify UI:** Confirm that the header, sidebar navigation, and model info render properly.
2. **Test Clinical Samples:**
   - Click **Sample: Cancer** $\rightarrow$ click **Analyze Image** $\rightarrow$ verify classification, Grad-CAM overlay, and PDF report download.
   - Click **Sample: Diverticula** $\rightarrow$ click **Analyze Image** $\rightarrow$ verify classification and Grad-CAM.
3. **Test OOD Gate:** Upload a non-endoscopic image (photo, document, screenshot) $\rightarrow$ verify that the domain verification gate rejects the image and suppresses inference.
4. **Test PDF Report:** Click **Download Analysis Report (PDF)** and verify the generated PDF.
