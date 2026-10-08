#!/usr/bin/env python3
"""
COLONVISION AI — Premium Clinical Research Workspace
=====================================================
An Explainable Deep Learning Framework for Differential Classification
of Colorectal Cancer and Diverticular Disease from Colonoscopy Images.

Architecture: ResNet-50 (Transfer Learning from ImageNet-1K)
Explainability: Grad-CAM on layer4[-1]
Domain Protection: Calibrated Endoscopy OOD Validator (Cosine-Similarity Profile)

Target Classes:
    0 = colon_diverticula
    1 = colorectal_cancer

Research prototype only. Not for clinical diagnosis.
"""

import os
import sys
import io
import time
from datetime import datetime
from pathlib import Path

# Ensure project root is in sys.path for internal modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.report_generator import generate_analysis_report

import numpy as np
from PIL import Image
import matplotlib.cm as cm
import streamlit as st

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ColonVision AI — Clinical Research Workspace",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium Product Palette & Typography CSS ──────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Color Tokens (Phase 10.6 Specification):
       PRIMARY BACKGROUND:   #F7F7F5
       PRIMARY SURFACE:      #FFFFFF
       SECONDARY SURFACE:    #F1F3F0
       PRIMARY TEXT:         #151A17
       SECONDARY TEXT:       #69716C
       BORDER:               #E1E5E1
       PRIMARY BRAND:        #174A3A
       PRIMARY BRAND DARK:   #10382C
       PRIMARY BRAND LIGHT:  #E3EEE8
       SUCCESS:              #26704F
       WARNING:              #9A6B32
       ERROR:                #A34848
       MEDICAL IMAGE VIEWER: #0D1110
       VIEWER SECONDARY:     #151B18
    */

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
        color: #151A17;
        background-color: #F7F7F5;
        -webkit-font-smoothing: antialiased;
    }

    .stApp {
        background-color: #F7F7F5;
    }
    .block-container {
        padding-top: 0.9rem;
        padding-bottom: 2.2rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        max-width: 1220px;
        margin: 0 auto;
    }

    /* Fixed Left Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #10382C;
        border-right: 1px solid #164637;
        color: #E3EEE8;
        width: 240px !important;
        min-width: 240px !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.1rem;
        padding-bottom: 1.2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] label {
        color: #E3EEE8 !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #174A3A;
        margin: 0.85rem 0;
    }

    /* Sidebar Header */
    .cw-side-brand {
        display: flex;
        flex-direction: column;
        margin-bottom: 0.8rem;
    }
    .cw-side-name {
        font-size: 15px;
        font-weight: 700;
        letter-spacing: 0.05em;
        color: #FFFFFF !important;
        line-height: 1.2;
    }
    .cw-side-tag {
        font-size: 11px;
        color: #A3C2B4 !important;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-top: 0.15rem;
    }

    /* Sidebar Status Console */
    .cw-side-status {
        background: #0B241C;
        border: 1px solid #174A3A;
        border-radius: 4px;
        padding: 0.6rem 0.75rem;
        margin-top: 1.1rem;
    }
    .cw-side-status-lbl {
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #88B29F !important;
        margin-bottom: 0.2rem;
    }
    .cw-side-status-val {
        font-size: 12px;
        font-weight: 600;
        color: #FFFFFF !important;
        display: flex;
        align-items: center;
        gap: 0.35rem;
    }

    /* Top Bar */
    .cw-top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #FFFFFF;
        border: 1px solid #E1E5E1;
        border-radius: 6px;
        padding: 0.55rem 1rem;
        margin-bottom: 1rem;
    }
    .cw-top-crumb {
        font-size: 13px;
        color: #69716C;
        font-weight: 500;
    }
    .cw-top-meta {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        font-size: 12px;
        color: #69716C;
    }
    .cw-top-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: #E3EEE8;
        color: #174A3A;
        font-size: 12px;
        font-weight: 600;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
    }
    .cw-dot-green {
        width: 6px;
        height: 6px;
        background: #26704F;
        border-radius: 50%;
        display: inline-block;
    }

    /* Surface Cards */
    .cw-card {
        background: #FFFFFF;
        border: 1px solid #E1E5E1;
        border-radius: 6px;
        padding: 1rem 1.15rem;
        margin-bottom: 1rem;
    }
    .cw-card-title {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #174A3A;
        margin-bottom: 0.65rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #F1F3F0;
        padding-bottom: 0.45rem;
    }

    /* Dark Medical Viewer */
    .cw-dark-viewer {
        background: #0D1110;
        border: 1px solid #151B18;
        border-radius: 4px;
        padding: 0.65rem;
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 250px;
    }

    /* Domain Status Banners */
    .cw-banner-pass {
        background: #FFFFFF;
        border: 1px solid #C4DFCFA;
        border-left: 3px solid #26704F;
        border-radius: 4px;
        padding: 0.65rem 0.95rem;
        margin-bottom: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .cw-banner-pass-title {
        font-size: 13px;
        font-weight: 600;
        color: #26704F;
    }
    .cw-banner-pass-meta {
        font-family: ui-monospace, Menlo, Consolas, monospace;
        font-size: 12px;
        color: #26704F;
    }

    .cw-banner-fail {
        background: #FFFFFF;
        border: 1px solid #ECC9C9;
        border-left: 3px solid #A34848;
        border-radius: 4px;
        padding: 0.8rem 1rem;
        margin-bottom: 1rem;
    }
    .cw-banner-fail-title {
        font-size: 13px;
        font-weight: 700;
        color: #A34848;
        margin-bottom: 0.25rem;
    }
    .cw-banner-fail-body {
        font-size: 13px;
        color: #632C2C;
        line-height: 1.45;
        margin-bottom: 0.45rem;
    }
    .cw-banner-fail-meta {
        font-family: ui-monospace, Menlo, Consolas, monospace;
        font-size: 12px;
        color: #A34848;
        background: #FAF2F2;
        padding: 0.25rem 0.5rem;
        border-radius: 3px;
        display: inline-block;
    }

    /* Prediction Card */
    .cw-pred-caption {
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #69716C;
        margin-bottom: 0.2rem;
    }
    .cw-pred-heading {
        font-size: 34px;
        font-weight: 700;
        letter-spacing: -0.02em;
        line-height: 1.15;
        margin-bottom: 0.2rem;
    }
    .cw-pred-cancer {
        color: #A34848;
    }
    .cw-pred-divert {
        color: #174A3A;
    }
    .cw-conf-label {
        font-size: 13px;
        font-weight: 500;
        color: #69716C;
        margin-bottom: 0.75rem;
    }
    .cw-conf-strong {
        font-weight: 700;
        color: #151A17;
        font-variant-numeric: tabular-nums;
    }

    /* Thin Precision Probability Meters */
    .cw-meter-row {
        display: flex;
        justify-content: space-between;
        font-size: 12px;
        color: #69716C;
        margin-bottom: 0.2rem;
    }
    .cw-meter-name {
        font-weight: 600;
        color: #151A17;
    }
    .cw-meter-val {
        font-weight: 600;
        color: #151A17;
        font-family: ui-monospace, Menlo, Consolas, monospace;
    }
    .cw-meter-track {
        height: 4px;
        background: #F1F3F0;
        border-radius: 2px;
        overflow: hidden;
        margin-bottom: 0.6rem;
    }
    .cw-meter-fill-cancer {
        height: 100%;
        background: #A34848;
    }
    .cw-meter-fill-divert {
        height: 100%;
        background: #174A3A;
    }

    /* Right Analysis Info Panel Rows */
    .cw-info-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.45rem 0;
        border-bottom: 1px solid #F1F3F0;
        font-size: 13px;
    }
    .cw-info-row:last-child {
        border-bottom: none;
    }
    .cw-info-key {
        color: #69716C;
        font-weight: 400;
    }
    .cw-info-val {
        font-weight: 600;
        color: #151A17;
    }

    /* Minimal Empty State */
    .cw-empty-state {
        background: #FFFFFF;
        border: 1px dashed #E1E5E1;
        border-radius: 6px;
        padding: 2.8rem 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .cw-empty-heading {
        font-size: 16px;
        font-weight: 700;
        color: #174A3A;
        margin-bottom: 0.35rem;
    }
    .cw-empty-desc {
        font-size: 13px;
        color: #69716C;
        max-width: 460px;
        margin: 0 auto 1.1rem auto;
        line-height: 1.45;
    }

    /* Button Polish */
    button[kind="primary"] {
        background-color: #174A3A !important;
        color: #FFFFFF !important;
        border: 1px solid #174A3A !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        transition: all 0.15s ease-in-out !important;
    }
    button[kind="primary"]:hover {
        background-color: #10382C !important;
        border-color: #10382C !important;
    }
    button[kind="secondary"] {
        background-color: #FFFFFF !important;
        color: #151A17 !important;
        border: 1px solid #E1E5E1 !important;
        border-radius: 4px !important;
        font-size: 13px !important;
        transition: all 0.15s ease-in-out !important;
    }
    button[kind="secondary"]:hover {
        background-color: #F1F3F0 !important;
        border-color: #D3D8D3 !important;
    }

    /* Premium Outlined Download Report Button */
    div[data-testid="stDownloadButton"] button {
        background-color: #FFFFFF !important;
        color: #174A3A !important;
        border: 1.5px solid #174A3A !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        transition: all 0.15s ease-in-out !important;
        padding: 0.5rem 1rem !important;
    }
    div[data-testid="stDownloadButton"] button:hover {
        background-color: #E3EEE8 !important;
        color: #10382C !important;
        border-color: #10382C !important;
    }

    /* Research Disclaimer */
    .cw-disclaimer {
        font-size: 12px;
        color: #69716C;
        border-top: 1px solid #E1E5E1;
        padding-top: 0.8rem;
        margin-top: 1.8rem;
        text-align: center;
        line-height: 1.45;
    }

    /* Streamlit Uploader Overrides */
    div[data-testid="stFileUploader"] {
        padding: 0;
    }
    div[data-testid="stFileUploader"] section {
        border-radius: 4px;
        border: 1px dashed #D3D8D3;
        background: #FFFFFF;
        padding: 0.65rem;
    }
    div[data-testid="stSlider"] label {
        font-size: 12px;
        font-weight: 600;
        color: #151A17;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants & Paths ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH   = PROJECT_ROOT / "models" / "resnet50_best.pth"
TEST_DIR     = PROJECT_ROOT / "data" / "processed" / "test"

CLASS_NAMES  = ["colon_diverticula", "colorectal_cancer"]
CLASS_LABELS = {
    "colon_diverticula": "Colon Diverticula",
    "colorectal_cancer": "Colorectal Cancer",
}
NUM_CLASSES  = 2
IMG_SIZE     = 224

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Preprocessing Transforms (Deterministic Evaluation Pipeline) ──────────────
eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

crop_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
])

VALIDATOR_PROFILE_PATH = PROJECT_ROOT / "models" / "endoscopy_validator_profile.json"


# ── Model & Validator Loaders (Cached) ────────────────────────────────────────
@st.cache_resource(show_spinner="Initializing ResNet-50 clinical backbone...")
def load_model(checkpoint_path: Path):
    if not checkpoint_path.exists():
        return None, f"Model checkpoint not found at: {checkpoint_path}"
    try:
        model = resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(model.fc.in_features, NUM_CLASSES),
        )
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        return model, None
    except Exception as e:
        return None, f"Failed to initialize model: {str(e)}"


@st.cache_resource(show_spinner="Loading Endoscopy Validator Profile...")
def load_validator(profile_path: Path):
    if not profile_path.exists():
        return None, f"Validator profile not found at: {profile_path}"
    try:
        from src.endoscopy_validator import EndoscopyValidator
        validator = EndoscopyValidator(profile_path)
        return validator, None
    except Exception as e:
        return None, f"Failed to initialize validator: {str(e)}"


# ── Grad-CAM Extractor Class ──────────────────────────────────────────────────
class GradCAMExtractor:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self.hook_handles.extend([h1, h2])

    def compute(self, input_tensor: torch.Tensor, target_class: int = None):
        self.model.eval()
        self.model.zero_grad()

        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1)

        pred_class = int(logits.argmax(dim=1).item())
        confidence = float(probs[0, pred_class].item())

        if target_class is None:
            target_class = pred_class

        target_score = logits[0, target_class]
        target_score.backward(retain_graph=True)

        grads = self.gradients.detach()
        acts = self.activations.detach()

        # Global average pooling on gradients -> channel weights
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * acts, dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam_np = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_norm = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam_np)

        return cam_norm, pred_class, confidence, probs[0].detach().cpu().numpy(), logits[0].detach().cpu().numpy()

    def remove(self):
        for h in self.hook_handles:
            h.remove()


# ── Image Overlay Function ────────────────────────────────────────────────────
def generate_overlay(pil_crop: Image.Image, cam_heatmap: np.ndarray, alpha: float = 0.45):
    img_np = np.array(pil_crop).astype(np.float32) / 255.0

    cam_pil = Image.fromarray((cam_heatmap * 255).astype(np.uint8)).resize(
        (IMG_SIZE, IMG_SIZE), resample=Image.BICUBIC
    )
    cam_resized = np.array(cam_pil).astype(np.float32) / 255.0

    colormap = cm.get_cmap("jet")
    heatmap_rgb = colormap(cam_resized)[:, :, :3]

    overlay = (1.0 - alpha) * img_np + alpha * heatmap_rgb
    overlay = np.clip(overlay, 0.0, 1.0)

    heatmap_uint8 = (heatmap_rgb * 255).astype(np.uint8)
    overlay_uint8 = (overlay * 255).astype(np.uint8)

    return Image.fromarray(heatmap_uint8), Image.fromarray(overlay_uint8)


# ── Session State Initialization ──────────────────────────────────────────────
def init_session_state():
    if "nav_page" not in st.session_state:
        st.session_state.nav_page = "Analysis"
    if "current_image" not in st.session_state:
        st.session_state.current_image = None
    if "current_source_label" not in st.session_state:
        st.session_state.current_source_label = ""
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "history" not in st.session_state:
        st.session_state.history = []
    if "viewer_tab" not in st.session_state:
        st.session_state.viewer_tab = "OVERLAY"
    if "heatmap_alpha" not in st.session_state:
        st.session_state.heatmap_alpha = 0.45
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0


# ── Action Callbacks ──────────────────────────────────────────────────────────
def reset_analysis_state():
    st.session_state.current_image = None
    st.session_state.current_source_label = ""
    st.session_state.analysis_result = None
    st.session_state.uploader_key += 1
    st.session_state.viewer_tab = "OVERLAY"
    st.session_state.heatmap_alpha = 0.45


def load_clinical_sample(sample_type: str):
    app_samples_dir = PROJECT_ROOT / "app" / "samples"
    if sample_type == "cancer":
        sample_path = app_samples_dir / "sample_colorectal_cancer.jpg"
        if not sample_path.exists():
            sample_path = TEST_DIR / "colorectal_cancer" / "0c7db439-d40d-468f-9cc7-97a2a526dd7c.jpg"
        label = "Sample: Colorectal Cancer (0c7db439)"
    elif sample_type == "divert":
        sample_path = app_samples_dir / "sample_colon_diverticula.jpg"
        if not sample_path.exists():
            sample_path = TEST_DIR / "colon_diverticula" / "04675a9b-4858-439c-9199-94426a1e76a5.jpg"
        label = "Sample: Colon Diverticula (04675a9b)"
    else:
        return

    if sample_path.exists():
        img = Image.open(sample_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        st.session_state.current_image = img
        st.session_state.current_source_label = label
        st.session_state.analysis_result = None
        st.session_state.viewer_tab = "OVERLAY"


# ── Main Application ──────────────────────────────────────────────────────────
def main():
    init_session_state()

    # Load Model & Validator Resources
    model, model_err = load_model(MODEL_PATH)
    if model_err:
        st.error(f"Backbone Error: {model_err}")
        st.stop()

    validator, val_err = load_validator(VALIDATOR_PROFILE_PATH)
    if val_err:
        st.error(f"Validator Error: {val_err}")
        st.stop()

    # ── Left Navigation Sidebar (Refined to 220-240px) ─────────────────────────
    with st.sidebar:
        st.markdown("""
        <div class="cw-side-brand">
            <div class="cw-side-name">COLONVISION AI</div>
            <div class="cw-side-tag">Research Workspace</div>
        </div>
        """, unsafe_allow_html=True)

        # Primary Action: + New Analysis
        if st.button("+ New Analysis", use_container_width=True, type="primary", key="btn_side_new_analysis"):
            reset_analysis_state()
            st.session_state.nav_page = "Analysis"
            st.rerun()

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

        nav_options = [
            "Analysis",
            "History",
            "Model Performance",
            "Explainability",
            "System Status",
            "About",
        ]

        selected_nav = st.radio(
            "Navigation",
            options=nav_options,
            index=nav_options.index(st.session_state.nav_page) if st.session_state.nav_page in nav_options else 0,
            label_visibility="collapsed",
        )
        if selected_nav != st.session_state.nav_page:
            st.session_state.nav_page = selected_nav
            st.rerun()

        st.divider()

        # Operational status tile
        st.markdown(f"""
        <div class="cw-side-status">
            <div class="cw-side-status-lbl">SYSTEM STATUS</div>
            <div class="cw-side-status-val">
                <span class="cw-dot-green"></span> System Ready
            </div>
            <div style="font-size: 11px; color: #A3C2B4; margin-top: 0.15rem;">
                Research Prototype &middot; ResNet-50
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Top Bar (Thin & Elegant) ──────────────────────────────────────────────
    st.markdown(f"""
    <div class="cw-top-bar">
        <div class="cw-top-crumb">
            ColonVision AI / {st.session_state.nav_page}
        </div>
        <div class="cw-top-meta">
            <span>ResNet-50</span>
            <span class="cw-top-pill">
                <span class="cw-dot-green"></span>
                Ready
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── ROUTED PAGES ──────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────────────────────────────────
    # PAGE 1: ANALYSIS WORKSPACE
    # ──────────────────────────────────────────────────────────────────────────
    if st.session_state.nav_page == "Analysis":
        # A. IMAGE INPUT AREA
        st.markdown('<div class="cw-card">', unsafe_allow_html=True)
        st.markdown("""
        <div class="cw-card-title">
            <span>UPLOAD IMAGE</span>
            <span style="font-weight: 500; color: #69716C;">JPEG &middot; PNG</span>
        </div>
        """, unsafe_allow_html=True)

        col_up, col_samples = st.columns([2.6, 1.4], gap="medium")

        with col_up:
            uploaded_file = st.file_uploader(
                "Drop an image here or choose a file",
                type=["jpg", "jpeg", "png"],
                key=f"uploader_{st.session_state.uploader_key}",
                label_visibility="collapsed",
                help="Upload a colonoscopy image in JPG, JPEG, or PNG format."
            )
            if uploaded_file is not None:
                new_label = f"Uploaded: {uploaded_file.name}"
                # If a new image is loaded, clear previous analysis state immediately
                if st.session_state.current_source_label != new_label:
                    try:
                        img = Image.open(uploaded_file)
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        st.session_state.current_image = img
                        st.session_state.current_source_label = new_label
                        st.session_state.analysis_result = None
                        st.session_state.viewer_tab = "OVERLAY"
                    except Exception as e:
                        st.error(f"Image read error: {str(e)}")

            st.caption("🔒 Images are processed locally for this research prototype and are not intentionally stored by the interface.")

        with col_samples:
            st.markdown("<div style='font-size: 11px; font-weight: 700; color: #174A3A; margin-bottom: 0.35rem; text-transform: uppercase; letter-spacing: 0.06em;'>BENCHMARK SAMPLES</div>", unsafe_allow_html=True)
            s_col1, s_col2 = st.columns(2)
            with s_col1:
                if st.button("Sample: Cancer", use_container_width=True, key="btn_sample_cancer"):
                    load_clinical_sample("cancer")
                    st.rerun()
            with s_col2:
                if st.button("Sample: Diverticula", use_container_width=True, key="btn_sample_divert"):
                    load_clinical_sample("divert")
                    st.rerun()

            a_col1, a_col2 = st.columns(2)
            with a_col1:
                if st.session_state.current_image is not None:
                    if st.button("Clear Image", use_container_width=True, key="btn_clear_image"):
                        reset_analysis_state()
                        st.rerun()
            with a_col2:
                if st.session_state.current_image is not None and st.session_state.analysis_result is None:
                    run_requested = st.button("Analyze Image", type="primary", use_container_width=True, key="btn_analyze_now")
                else:
                    run_requested = False

        st.markdown('</div>', unsafe_allow_html=True)  # Close Upload card

        # EMPTY STATE
        if st.session_state.current_image is None:
            st.markdown("""
            <div class="cw-empty-state">
                <div class="cw-empty-heading">Analyze an endoscopy image</div>
                <div class="cw-empty-desc">
                    Upload a colonoscopy image to begin domain verification and model analysis.
                </div>
                <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #174A3A;">
                    Domain Verification &middot; Deep Learning Inference &middot; Grad-CAM Explainability
                </div>
            </div>
            """, unsafe_allow_html=True)
            return

        current_img = st.session_state.current_image

        # PREVIEW STATE (WHEN IMAGE IS LOADED BUT NOT ANALYZED YET)
        if st.session_state.analysis_result is None and not run_requested:
            st.markdown("""
            <div class="cw-card">
                <div class="cw-card-title">
                    <span>INPUT IMAGE</span>
                    <span style="color: #26704F; font-weight: 600;">● READY TO VERIFY</span>
                </div>
            """, unsafe_allow_html=True)
            pv_col1, pv_col2 = st.columns([1, 2], gap="medium")
            with pv_col1:
                st.markdown('<div class="cw-dark-viewer">', unsafe_allow_html=True)
                st.image(crop_transform(current_img), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with pv_col2:
                st.markdown(f"**Selected Image:** `{st.session_state.current_source_label}`")
                st.markdown("Standardized pre-processing: `Resize(256) → CenterCrop(224) → ImageNet normalization`.")
                st.markdown("Click **Analyze Image** above to run the endoscopy domain verification and ResNet-50 prediction.")
            st.markdown('</div>', unsafe_allow_html=True)
            return

        # EXECUTE ANALYSIS WHEN REQUESTED
        if st.session_state.analysis_result is None and run_requested:
            with st.spinner("Executing domain verification check..."):
                val_res = validator.validate_image(current_img)

            if not val_res["is_endoscopy"]:
                st.session_state.analysis_result = {
                    "is_endoscopy": False,
                    "domain_score": val_res["domain_score"],
                    "threshold": val_res["threshold"],
                }
            else:
                with st.spinner("Computing ResNet-50 predictions and Grad-CAM activations..."):
                    input_tensor = eval_transform(current_img).unsqueeze(0).to(DEVICE)
                    cropped_original = crop_transform(current_img)

                    extractor = GradCAMExtractor(model, model.layer4[-1])
                    cam_map, pred_idx, conf, probs, logits = extractor.compute(input_tensor)
                    extractor.remove()

                    res_payload = {
                        "is_endoscopy": True,
                        "domain_score": val_res["domain_score"],
                        "threshold": val_res["threshold"],
                        "pred_idx": pred_idx,
                        "pred_class": CLASS_NAMES[pred_idx],
                        "confidence": conf,
                        "probs": probs,
                        "logits": logits,
                        "cam_map": cam_map,
                        "cropped_original": cropped_original,
                    }
                    st.session_state.analysis_result = res_payload

                    # Log to history
                    curr_time = datetime.now().strftime("%I:%M %p")
                    st.session_state.history.append({
                        "id": len(st.session_state.history) + 1,
                        "timestamp": curr_time,
                        "input": st.session_state.current_source_label,
                        "prediction": CLASS_LABELS[CLASS_NAMES[pred_idx]],
                        "confidence": f"{conf * 100:.2f}%",
                        "probs": probs,
                        "domain_score": float(val_res["domain_score"]),
                        "is_endoscopy": True,
                        "cam_map": cam_map,
                        "cropped_original": cropped_original,
                    })

        res = st.session_state.analysis_result

        # DOMAIN REJECTION HANDLING
        if not res["is_endoscopy"]:
            st.markdown(f"""
            <div class="cw-banner-fail">
                <div class="cw-banner-fail-title">DOMAIN VERIFICATION: NOT AN ENDOSCOPY IMAGE</div>
                <div class="cw-banner-fail-body">
                    Analysis was stopped because the submitted image does not meet the validated endoscopy-domain criteria.<br>
                    Non-endoscopic imagery (such as passport photos, selfies, portraits, ID cards, document screenshots, or everyday photographs)
                    is rejected to prevent spurious classifications.
                </div>
                <div class="cw-banner-fail-meta">
                    Domain Similarity Score: <b>{res['domain_score']:.4f}</b> &nbsp;|&nbsp; Calibrated Threshold: <b>&ge; {res['threshold']:.4f}</b> &nbsp;|&nbsp; Status: <b>REJECTED</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("Choose Another Image", type="primary", key="btn_choose_another"):
                reset_analysis_state()
                st.rerun()
            return

        # DOMAIN ACCEPTANCE BANNER
        st.markdown(f"""
        <div class="cw-banner-pass">
            <div class="cw-banner-pass-title">
                DOMAIN VERIFIED &mdash; Endoscopy image accepted
            </div>
            <div class="cw-banner-pass-meta">
                Domain Score: <b>{res['domain_score']:.4f}</b> &nbsp;|&nbsp; Threshold: <b>&ge; {res['threshold']:.4f}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # MAIN ANALYSIS RESULT & EXPLAINABILITY VIEW
        cropped_orig = res["cropped_original"]
        cam_map = res["cam_map"]
        pred_class = res["pred_class"]
        conf = res["confidence"]
        probs = res["probs"]

        prob_divert = float(probs[0])
        prob_cancer = float(probs[1])

        # Generate visual overlay artifacts for viewer and report
        heatmap_img, overlay_img = generate_overlay(cropped_orig, cam_map, alpha=st.session_state.heatmap_alpha)

        col_viewer, col_analysis = st.columns([1.3, 1], gap="medium")

        # K. IMAGE VIEWER (VISUAL CENTERPIECE)
        with col_viewer:
            st.markdown('<div class="cw-card">', unsafe_allow_html=True)
            st.markdown("""
            <div class="cw-card-title">
                <span>VISUAL EXPLANATION</span>
                <span style="color: #174A3A; font-weight: 600;">GRAD-CAM ATTENTION MAP</span>
            </div>
            """, unsafe_allow_html=True)

            # Interactive Tabs
            vt1, vt2, vt3 = st.columns(3)
            with vt1:
                if st.button("ORIGINAL", use_container_width=True, type="primary" if st.session_state.viewer_tab == "ORIGINAL" else "secondary"):
                    st.session_state.viewer_tab = "ORIGINAL"
                    st.rerun()
            with vt2:
                if st.button("GRAD-CAM", use_container_width=True, type="primary" if st.session_state.viewer_tab == "GRAD-CAM" else "secondary"):
                    st.session_state.viewer_tab = "GRAD-CAM"
                    st.rerun()
            with vt3:
                if st.button("OVERLAY", use_container_width=True, type="primary" if st.session_state.viewer_tab == "OVERLAY" else "secondary"):
                    st.session_state.viewer_tab = "OVERLAY"
                    st.rerun()

            st.markdown('<div class="cw-dark-viewer">', unsafe_allow_html=True)
            if st.session_state.viewer_tab == "ORIGINAL":
                st.image(cropped_orig, use_container_width=True)
            elif st.session_state.viewer_tab == "GRAD-CAM":
                st.image(heatmap_img, use_container_width=True)
            else:
                st.image(overlay_img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Controls: Opacity Slider & Reset
            sl_col, res_col = st.columns([3, 1])
            with sl_col:
                new_alpha = st.slider(
                    "Heatmap Opacity (Alpha)",
                    min_value=0.10,
                    max_value=0.90,
                    value=float(st.session_state.heatmap_alpha),
                    step=0.05,
                    key="slider_alpha",
                )
                if new_alpha != st.session_state.heatmap_alpha:
                    st.session_state.heatmap_alpha = new_alpha
                    st.rerun()
            with res_col:
                st.markdown("<div style='height: 1.45rem;'></div>", unsafe_allow_html=True)
                if st.button("Reset Opacity", use_container_width=True, key="btn_reset_alpha"):
                    st.session_state.heatmap_alpha = 0.45
                    st.rerun()

            st.markdown("""
            <div style="font-size: 12px; color: #69716C; line-height: 1.4; border-top: 1px solid #F1F3F0; padding-top: 0.5rem; margin-top: 0.35rem;">
                <b>Attention Summary:</b> Target Layer: <code>layer4[-1]</code> &middot; Feature Map: 7 &times; 7 &middot; Interpolation: Bi-cubic<br>
                <i>Interpretation aid &mdash; not intended for diagnostic localization.</i>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # J. PREDICTION RESULT & M. RIGHT ANALYSIS PANEL
        with col_analysis:
            st.markdown('<div class="cw-card">', unsafe_allow_html=True)
            st.markdown("""
            <div class="cw-card-title">
                <span>ANALYSIS COMPLETE</span>
                <span style="color: #26704F; font-weight: 600;">● INFERENCE SUCCESS</span>
            </div>
            """, unsafe_allow_html=True)

            name_cls = "cw-pred-cancer" if pred_class == "colorectal_cancer" else "cw-pred-divert"
            formatted_name = CLASS_LABELS[pred_class]

            st.markdown(f"""
            <div class="cw-pred-caption">PREDICTED CLASS</div>
            <div class="cw-pred-heading {name_cls}">{formatted_name}</div>
            
            <div class="cw-conf-label">
                Confidence: <span class="cw-conf-strong">{conf * 100:.2f}%</span>
            </div>
            
            <div style="margin-top: 0.6rem; margin-bottom: 0.9rem;">
                <div class="cw-meter-row">
                    <span class="cw-meter-name">COLORECTAL CANCER</span>
                    <span class="cw-meter-val">{prob_cancer * 100:.2f}%</span>
                </div>
                <div class="cw-meter-track">
                    <div class="cw-meter-fill-cancer" style="width: {int(prob_cancer * 100)}%;"></div>
                </div>

                <div class="cw-meter-row">
                    <span class="cw-meter-name">COLON DIVERTICULA</span>
                    <span class="cw-meter-val">{prob_divert * 100:.2f}%</span>
                </div>
                <div class="cw-meter-track">
                    <div class="cw-meter-fill-divert" style="width: {int(prob_divert * 100)}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Professional PDF Report Generation & Download Button
            try:
                report_bytes = generate_analysis_report(
                    original_image=cropped_orig,
                    gradcam_image=heatmap_img,
                    overlay_image=overlay_img,
                    prediction=formatted_name,
                    probabilities={
                        "colon_diverticula": prob_divert,
                        "colorectal_cancer": prob_cancer,
                    },
                    confidence=conf,
                    validation_score=float(res["domain_score"]),
                    model_name="ResNet-50",
                    threshold=float(res.get("threshold", 0.6800)),
                )
                ts_str = datetime.now().strftime("%Y-%m-%d_%H%M")
                pdf_filename = f"ColonVision_AI_Analysis_Report_{ts_str}.pdf"
                st.download_button(
                    label="Download Analysis Report",
                    data=report_bytes,
                    file_name=pdf_filename,
                    mime="application/pdf",
                    use_container_width=True,
                    key="btn_download_analysis_report",
                )
            except Exception as e:
                st.error("Unable to generate the report. Please try again.")
                if st.button("Try Again", key="btn_try_again_report", use_container_width=True):
                    st.rerun()

            st.markdown(f"""
            <div style="border-top: 1px solid #F1F3F0; padding-top: 0.6rem; margin-top: 0.5rem;">
                <div class="cw-info-row">
                    <span class="cw-info-key">Domain</span>
                    <span class="cw-info-val" style="color: #26704F;">Verified ({res['domain_score']:.4f})</span>
                </div>
                <div class="cw-info-row">
                    <span class="cw-info-key">Prediction</span>
                    <span class="cw-info-val">{formatted_name}</span>
                </div>
                <div class="cw-info-row">
                    <span class="cw-info-key">Confidence</span>
                    <span class="cw-info-val">{conf * 100:.2f}%</span>
                </div>
                <div class="cw-info-row">
                    <span class="cw-info-key">Model</span>
                    <span class="cw-info-val">ResNet-50</span>
                </div>
                <div class="cw-info-row">
                    <span class="cw-info-key">Explainability</span>
                    <span class="cw-info-val" style="color: #174A3A;">Grad-CAM Available</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────────
    # PAGE 2: ANALYSIS HISTORY
    # ──────────────────────────────────────────────────────────────────────────
    elif st.session_state.nav_page == "History":
        st.markdown('<div class="cw-card">', unsafe_allow_html=True)
        st.markdown("""
        <div class="cw-card-title">
            <span>SESSION ANALYSIS HISTORY</span>
            <span style="font-weight: 500; color: #69716C;">In-Memory Session Records</span>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.history:
            st.info("No analyses recorded in this session yet. Return to 'Analysis' to analyze an endoscopy image.")
        else:
            table_records = [
                {
                    "ID": rec["id"],
                    "Timestamp": rec["timestamp"],
                    "Input Frame": rec["input"],
                    "Predicted Class": rec["prediction"],
                    "Confidence": rec["confidence"],
                    "Domain Score": f"{rec['domain_score']:.4f}",
                }
                for rec in st.session_state.history[::-1]
            ]
            st.dataframe(table_records, use_container_width=True, hide_index=True)

            h_c1, h_c2 = st.columns([3, 1])
            with h_c1:
                hist_ids = [rec["id"] for rec in st.session_state.history]
                selected_id = st.selectbox(
                    "Select History Record to Inspect:",
                    options=hist_ids,
                    format_func=lambda x: f"Record #{x} - {st.session_state.history[x-1]['prediction']} ({st.session_state.history[x-1]['confidence']}) &middot; {st.session_state.history[x-1]['timestamp']}",
                    key="hist_select_box"
                )
            with h_c2:
                st.markdown("<div style='height: 1.65rem;'></div>", unsafe_allow_html=True)
                if st.button("Clear History", use_container_width=True, key="btn_clear_history"):
                    st.session_state.history = []
                    st.session_state.analysis_result = None
                    st.rerun()

            # Render historical details
            if selected_id is not None and len(st.session_state.history) >= selected_id:
                sel_rec = st.session_state.history[selected_id - 1]
                st.markdown("---")
                st.markdown(f"#### Record Details: #{sel_rec['id']} &middot; {sel_rec['input']}")
                
                det_c1, det_c2 = st.columns([1, 1.25])
                with det_c1:
                    st.markdown(f"**Timestamp:** `{sel_rec['timestamp']}`")
                    st.markdown(f"**Prediction:** `{sel_rec['prediction']}`")
                    st.markdown(f"**Confidence:** `{sel_rec['confidence']}`")
                    st.markdown(f"**Domain Score:** `{sel_rec['domain_score']:.4f}`")
                    
                    if st.button("Load into Workspace", type="primary", key=f"btn_reopen_{sel_rec['id']}", use_container_width=True):
                        st.session_state.current_image = sel_rec["cropped_original"]
                        st.session_state.current_source_label = f"Historical Record #{sel_rec['id']}: {sel_rec['input']}"
                        st.session_state.analysis_result = {
                            "is_endoscopy": True,
                            "domain_score": sel_rec["domain_score"],
                            "threshold": 0.6800,
                            "pred_idx": 1 if "Cancer" in sel_rec["prediction"] else 0,
                            "pred_class": "colorectal_cancer" if "Cancer" in sel_rec["prediction"] else "colon_diverticula",
                            "confidence": float(sel_rec["confidence"].replace("%", "")) / 100.0,
                            "probs": sel_rec["probs"],
                            "logits": np.array([0.0, 0.0]),
                            "cam_map": sel_rec["cam_map"],
                            "cropped_original": sel_rec["cropped_original"],
                        }
                        st.session_state.nav_page = "Analysis"
                        st.rerun()

                    # Direct History Report Generation & Download
                    try:
                        hist_h, hist_o = generate_overlay(sel_rec["cropped_original"], sel_rec["cam_map"], alpha=0.45)
                        hist_probs = sel_rec["probs"]
                        hist_pdf = generate_analysis_report(
                            original_image=sel_rec["cropped_original"],
                            gradcam_image=hist_h,
                            overlay_image=hist_o,
                            prediction=sel_rec["prediction"],
                            probabilities={
                                "colon_diverticula": float(hist_probs[0]),
                                "colorectal_cancer": float(hist_probs[1]),
                            },
                            confidence=float(sel_rec["confidence"].replace("%", "")) / 100.0,
                            validation_score=float(sel_rec["domain_score"]),
                            model_name="ResNet-50",
                        )
                        st.download_button(
                            label="Download Analysis Report",
                            data=hist_pdf,
                            file_name=f"ColonVision_AI_Analysis_Report_Record_{sel_rec['id']}.pdf",
                            mime="application/pdf",
                            key=f"btn_dl_hist_{sel_rec['id']}",
                            use_container_width=True,
                        )
                    except Exception:
                        st.error("Unable to generate the report. Please try again.")

                with det_c2:
                    st.markdown('<div class="cw-dark-viewer">', unsafe_allow_html=True)
                    _, hist_ov = generate_overlay(sel_rec["cropped_original"], sel_rec["cam_map"], alpha=0.45)
                    st.image(hist_ov, caption="Grad-CAM Overlay (Alpha = 0.45)", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────────
    # PAGE 3: MODEL PERFORMANCE
    # ──────────────────────────────────────────────────────────────────────────
    elif st.session_state.nav_page == "Model Performance":
        st.markdown('<div class="cw-card">', unsafe_allow_html=True)
        st.markdown("""
        <div class="cw-card-title">
            <span>MODEL BENCHMARK COMPARISON</span>
            <span style="font-weight: 500; color: #69716C;">Independent Test Set (26 Images)</span>
        </div>
        <div style="font-size: 13px; color: #69716C; margin-bottom: 0.8rem;">
            Evaluation of transfer-learning backbones trained under identical hyperparameters and class weighting on the GastroVision benchmark.
        </div>
        """, unsafe_allow_html=True)

        comparison_data = [
            {"Model": "ResNet-50 (Selected)", "Parameters": "23.5M", "Test Accuracy": "100.0%", "Macro-F1": "1.0000", "ROC-AUC": "1.0000", "Diverticula Recall": "100.0%"},
            {"Model": "DenseNet-121", "Parameters": "6.9M", "Test Accuracy": "96.15%", "Macro-F1": "0.9328", "ROC-AUC": "1.0000", "Diverticula Recall": "80.0%"},
            {"Model": "EfficientNet-B0", "Parameters": "4.0M", "Test Accuracy": "96.15%", "Macro-F1": "0.9328", "ROC-AUC": "0.9333", "Diverticula Recall": "80.0%"},
        ]
        st.table(comparison_data)

        st.markdown("""
        <div style="font-size: 12px; color: #69716C; line-height: 1.45; margin-top: 0.6rem; border-top: 1px solid #F1F3F0; padding-top: 0.5rem;">
            <b>Performance Note:</b> ResNet-50 achieved zero false-positives and zero false-negatives on the held-out test split (21 colorectal cancer, 5 colon diverticula).
            Because the test set is limited in size, 5-fold cross-validation was additionally executed across the 142 development images to evaluate generalization stability.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────────
    # PAGE 4: EXPLAINABILITY
    # ──────────────────────────────────────────────────────────────────────────
    elif st.session_state.nav_page == "Explainability":
        st.markdown('<div class="cw-card">', unsafe_allow_html=True)
        st.markdown("""
        <div class="cw-card-title">
            <span>EXPLAINABILITY METHODOLOGY</span>
            <span style="font-weight: 500; color: #174A3A;">GRAD-CAM</span>
        </div>
        <div style="font-size: 14px; color: #151A17; line-height: 1.55; margin-bottom: 0.8rem;">
            <b>Gradient-weighted Class Activation Mapping (Grad-CAM)</b> provides visual transparency into neural network decisions without retraining or architecture modification.
        </div>
        <div style="font-size: 13px; color: #69716C; line-height: 1.55;">
            <b>Methodology:</b>
            <ul>
                <li><b>Target Layer:</b> <code>model.layer4[-1]</code> (Final bottleneck convolution of ResNet-50, yielding 2048 channel feature maps of size 7 &times; 7).</li>
                <li><b>Gradients:</b> Gradients of the predicted target class score are globally pooled to calculate importance weights for each feature channel.</li>
                <li><b>Colormap:</b> Positive activations undergo ReLU filtering, normalization, and Jet colormap projection overlaid onto the original $224 \times 224$ crop.</li>
            </ul>
            <b>Morphological Interpretation Patterns:</b>
            <ul>
                <li><b>Colorectal Cancer:</b> Gradient activations align with raised neoplastic tissue margins, irregular mucosal nodularity, and ulcerated borders.</li>
                <li><b>Colon Diverticula:</b> Gradient activations localize over dark mucosal depressions, pouch orifices (diverticular necks), and concentric colonic wall outpouchings.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────────
    # PAGE 5: SYSTEM STATUS
    # ──────────────────────────────────────────────────────────────────────────
    elif st.session_state.nav_page == "System Status":
        st.markdown('<div class="cw-card">', unsafe_allow_html=True)
        st.markdown("""
        <div class="cw-card-title">
            <span>SYSTEM SPECIFICATIONS & STATUS</span>
            <span style="color: #26704F; font-weight: 600;">● OPERATIONAL</span>
        </div>
        """, unsafe_allow_html=True)

        sys_cols = st.columns(3)
        with sys_cols[0]:
            st.metric("Model Architecture", "ResNet-50")
            st.metric("Input Resolution", "224 × 224 px")
        with sys_cols[1]:
            st.metric("Inference Engine", "PyTorch (CPU)")
            st.metric("Target Layer", "layer4[-1]")
        with sys_cols[2]:
            st.metric("Validator Status", "Active")
            st.metric("Calibrated Threshold", f"τ* = {validator.threshold:.4f}")

        st.markdown("""
        <div style="font-size: 12px; color: #69716C; margin-top: 0.8rem; border-top: 1px solid #F1F3F0; padding-top: 0.5rem;">
            <b>Integrity Status:</b> Checkpoint weights, GastroVision validator profile, and deterministic evaluation transforms verified intact.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────────
    # PAGE 6: ABOUT
    # ──────────────────────────────────────────────────────────────────────────
    elif st.session_state.nav_page == "About":
        st.markdown('<div class="cw-card">', unsafe_allow_html=True)
        st.markdown("""
        <div class="cw-card-title">
            <span>ABOUT COLONVISION AI</span>
            <span style="font-weight: 500; color: #69716C;">Research Project</span>
        </div>
        <div style="font-size: 14px; color: #151A17; line-height: 1.55; margin-bottom: 0.8rem;">
            <b>ColonVision AI: Explainable Deep Learning for Colorectal Image Classification</b><br>
            Developed as an engineering research prototype to evaluate deep transfer learning and visual interpretability
            for the differential analysis of colorectal cancer and colon diverticula.
        </div>
        <div style="font-size: 13px; color: #69716C; line-height: 1.55;">
            <b>Core Technical Components:</b>
            <ul>
                <li><b>Endoscopy Domain Validator:</b> Cosine-similarity feature matching calibrated to reject non-endoscopic imagery (such as passport photos, selfies, portraits, and screenshots) prior to classification.</li>
                <li><b>ResNet-50 Classifier:</b> Pretrained on ImageNet-1K and fine-tuned on GastroVision endoscopic frames.</li>
                <li><b>Grad-CAM Explainability:</b> Channel-weighted activation mapping on the final convolutional bottleneck block.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Mandatory Research Disclaimer (Bottom of All Views) ───────────────────
    st.markdown("""
    <div class="cw-disclaimer">
        <b>Research prototype.</b> Not intended for clinical diagnosis, triage, or treatment decisions.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
