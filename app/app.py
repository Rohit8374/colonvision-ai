#!/usr/bin/env python3
"""
COLONVISION AI — Premium Clinical Research Workstation
======================================================
An Explainable Deep Learning Framework for Differential Classification
of Colorectal Cancer and Diverticular Disease from Colonoscopy Images.

Architectures:
    ResNet-50 (Recommended — best study benchmark)
    DenseNet-121
    EfficientNet-B0

Explainability: Grad-CAM (architecture-specific final convolutional block)
Domain Protection: Calibrated Endoscopy OOD Validator (Cosine-Similarity Profile)

Target Classes:
    0 = colon_diverticula
    1 = colorectal_cancer

Research prototype only. Not for clinical diagnosis.
"""

import sys
from datetime import datetime
from pathlib import Path

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
from torchvision.models import resnet50, densenet121, efficientnet_b0

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ColonVision AI — Clinical Research Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium Clinical Workstation Theme ────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    :root {
        --cv-bg: #F8F9F8;              /* warm ivory / off-white */
        --cv-surface: #FFFFFF;         /* clean white */
        --cv-surface-muted: #EFF3F0;   /* subtle muted clinical tint */
        --cv-text: #111816;            /* deep charcoal / near-black */
        --cv-text-sec: #43544C;        /* medium slate gray (high contrast >5.8:1) */
        --cv-border: #D5DDD8;          /* crisp thin border */
        --cv-border-light: #E4EBE7;    /* light divider */
        --cv-primary: #154536;         /* deep forest green */
        --cv-primary-dark: #0D2D23;    /* very dark forest green */
        --cv-primary-light: #E1EBE5;   /* refined forest tint */
        --cv-sage: #3D705C;            /* muted sage / emerald accent */
        --cv-success: #176346;         /* medical green */
        --cv-warning: #855514;         /* dark amber */
        --cv-error: #932828;           /* medical crimson */
        --cv-viewer: #0E1210;          /* very dark charcoal / black */
        --cv-viewer-alt: #161D1A;      /* dark charcoal border */
    }

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: var(--cv-text);
        background-color: var(--cv-bg);
        -webkit-font-smoothing: antialiased;
    }

    .stApp { background-color: var(--cv-bg); }

    .block-container {
        padding-top: 0.75rem;
        padding-bottom: 2rem;
        padding-left: 1.35rem;
        padding-right: 1.35rem;
        max-width: 1280px;
        margin: 0 auto;
    }

    /* ── Slim Left Sidebar ── */
    section[data-testid="stSidebar"] {
        background-color: var(--cv-primary-dark);
        border-right: 1px solid #1A4538;
        color: #E4EDE8;
        width: 220px !important;
        min-width: 220px !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 0.85rem;
        padding-bottom: 1rem;
        padding-left: 0.85rem;
        padding-right: 0.85rem;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: #D5E4DC;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #1A4538;
        margin: 0.7rem 0;
    }

    /* ── Sidebar Navigation Buttons ── */
    section[data-testid="stSidebar"] div[class*="st-key-nav_"] {
        margin-bottom: 0.2rem !important;
    }

    /* Inactive navigation items */
    section[data-testid="stSidebar"] div[class*="st-key-nav_"] button {
        background-color: transparent !important;
        border: 1px solid transparent !important;
        border-left: 3px solid transparent !important;
        border-radius: 4px !important;
        color: #E2ECE6 !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 0.45rem 0.75rem !important;
        min-height: 2.2rem !important;
        box-shadow: none !important;
        transition: background-color 0.12s ease, border-color 0.12s ease, color 0.12s ease !important;
    }

    section[data-testid="stSidebar"] div[class*="st-key-nav_"] button *,
    section[data-testid="stSidebar"] div[class*="st-key-nav_"] button p,
    section[data-testid="stSidebar"] div[class*="st-key-nav_"] button span,
    section[data-testid="stSidebar"] div[class*="st-key-nav_"] button div {
        color: #E2ECE6 !important;
        font-weight: 500 !important;
        font-size: 13px !important;
        text-align: left !important;
        justify-content: flex-start !important;
    }

    /* Inactive hover state */
    section[data-testid="stSidebar"] div[class*="st-key-nav_"] button:hover {
        background-color: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-left: 3px solid #3E7A66 !important;
        color: #FFFFFF !important;
    }

    section[data-testid="stSidebar"] div[class*="st-key-nav_"] button:hover *,
    section[data-testid="stSidebar"] div[class*="st-key-nav_"] button:hover p,
    section[data-testid="stSidebar"] div[class*="st-key-nav_"] button:hover span,
    section[data-testid="stSidebar"] div[class*="st-key-nav_"] button:hover div {
        color: #FFFFFF !important;
    }

    /* Active navigation item (type="primary") */
    section[data-testid="stSidebar"] div[class*="st-key-nav_"] button[kind="primary"] {
        background-color: #1E5040 !important;
        border: 1px solid #296854 !important;
        border-left: 3px solid #57B693 !important;
        border-radius: 4px !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }

    section[data-testid="stSidebar"] div[class*="st-key-nav_"] button[kind="primary"] *,
    section[data-testid="stSidebar"] div[class*="st-key-nav_"] button[kind="primary"] p,
    section[data-testid="stSidebar"] div[class*="st-key-nav_"] button[kind="primary"] span,
    section[data-testid="stSidebar"] div[class*="st-key-nav_"] button[kind="primary"] div {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    section[data-testid="stSidebar"] div[class*="st-key-nav_"] button[kind="primary"]:hover {
        background-color: #245E4C !important;
        border-color: #317A63 !important;
        border-left: 3px solid #57B693 !important;
        color: #FFFFFF !important;
    }

    /* "New Analysis" Action Button in Sidebar (Keep existing styling) */
    section[data-testid="stSidebar"] div[class*="st-key-btn_side_new_analysis"] {
        margin-bottom: 0.85rem !important;
    }

    section[data-testid="stSidebar"] div[class*="st-key-btn_side_new_analysis"] button {
        background-color: var(--cv-primary) !important;
        color: #FFFFFF !important;
        border: 1px solid var(--cv-primary) !important;
        border-radius: 3px !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        min-height: 2.35rem !important;
        justify-content: center !important;
        text-align: center !important;
    }

    section[data-testid="stSidebar"] div[class*="st-key-btn_side_new_analysis"] button *,
    section[data-testid="stSidebar"] div[class*="st-key-btn_side_new_analysis"] button p,
    section[data-testid="stSidebar"] div[class*="st-key-btn_side_new_analysis"] button span,
    section[data-testid="stSidebar"] div[class*="st-key-btn_side_new_analysis"] button div {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        justify-content: center !important;
    }

    section[data-testid="stSidebar"] div[class*="st-key-btn_side_new_analysis"] button:hover {
        background-color: #143D31 !important;
        border-color: #143D31 !important;
    }

    .cv-side-brand { margin-bottom: 0.85rem; }
    .cv-side-name {
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.12em;
        color: #FFFFFF !important;
        line-height: 1.25;
        text-transform: uppercase;
    }
    .cv-side-ai {
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.18em;
        color: #A8C4B6 !important;
        line-height: 1.25;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }
    .cv-side-tag {
        font-size: 10px;
        color: #8FA99A !important;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        font-weight: 500;
    }
    .cv-nav-group {
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #6F8F7E !important;
        margin: 0.85rem 0 0.35rem 0;
    }
    .cv-side-status {
        background: #0C241C;
        border: 1px solid #1A4538;
        border-radius: 3px;
        padding: 0.55rem 0.7rem;
        margin-top: 1rem;
    }
    .cv-side-status-val {
        font-size: 11px;
        font-weight: 600;
        color: #FFFFFF !important;
        letter-spacing: 0.04em;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* ── Top Header ── */
    .cv-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: var(--cv-surface);
        border: 1px solid var(--cv-border);
        border-radius: 4px;
        padding: 0.6rem 1.15rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(17, 24, 22, 0.03);
    }
    .cv-header-left { display: flex; flex-direction: column; gap: 0.15rem; }
    .cv-header-brand {
        font-size: 13.5px;
        font-weight: 700;
        letter-spacing: 0.1em;
        color: var(--cv-primary);
        text-transform: uppercase;
    }
    .cv-header-sub {
        font-size: 11px;
        color: var(--cv-text-sec);
        font-weight: 500;
    }
    .cv-header-right {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-size: 12px;
        color: var(--cv-text-sec);
    }
    .cv-header-tag {
        font-size: 11.5px;
        font-weight: 500;
        color: var(--cv-text-sec);
        letter-spacing: 0.02em;
    }
    .cv-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: var(--cv-primary-light);
        color: var(--cv-primary);
        font-size: 11px;
        font-weight: 600;
        padding: 0.22rem 0.6rem;
        border-radius: 3px;
        border: 1px solid #CAD6CF;
        letter-spacing: 0.03em;
    }
    .cv-model-pill {
        display: inline-flex;
        align-items: center;
        background: var(--cv-surface-muted);
        color: var(--cv-text);
        font-size: 11px;
        font-weight: 600;
        padding: 0.22rem 0.6rem;
        border-radius: 3px;
        border: 1px solid var(--cv-border);
        font-family: 'IBM Plex Mono', ui-monospace, monospace;
        letter-spacing: 0.02em;
    }
    .cv-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        display: inline-block;
        flex-shrink: 0;
    }
    .cv-dot-ok { background: var(--cv-success); }
    .cv-dot-err { background: var(--cv-error); }

    /* ── Cards & Panels ── */
    .cv-card {
        background: var(--cv-surface);
        border: 1px solid var(--cv-border);
        border-radius: 4px;
        padding: 0.95rem 1.1rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 1px 3px rgba(17, 24, 22, 0.03);
    }
    .cv-section-label {
        font-size: 11.5px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--cv-primary) !important;
        margin-bottom: 0.35rem;
    }
    .cv-section-desc {
        font-size: 13px;
        color: var(--cv-text-sec);
        line-height: 1.45;
        margin-bottom: 0.75rem;
    }
    .cv-card-title {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: var(--cv-primary) !important;
        margin-bottom: 0.65rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid var(--cv-surface-muted);
        padding-bottom: 0.4rem;
    }
    .cv-card-title span {
        color: var(--cv-primary) !important;
    }
    .cv-micro {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--cv-primary) !important;
    }

    /* ── Dark Medical Viewer ── */
    .cv-viewer {
        background: var(--cv-viewer);
        border: 1px solid var(--cv-viewer-alt);
        border-radius: 3px;
        padding: 0.75rem;
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 280px;
    }
    .cv-viewer-empty {
        text-align: center;
        padding: 2.5rem 1rem;
    }
    .cv-viewer-empty-title {
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #D5DDD8;
        margin-bottom: 0.45rem;
    }
    .cv-viewer-empty-desc {
        font-size: 13px;
        color: #9AB2A6;
        line-height: 1.4;
    }

    /* ── Domain Banners ── */
    .cv-banner-pass {
        background: var(--cv-surface);
        border: 1px solid #C5D9CE;
        border-left: 3px solid var(--cv-success);
        border-radius: 2px;
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.85rem;
    }
    .cv-banner-pass-title {
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--cv-success);
        margin-bottom: 0.15rem;
    }
    .cv-banner-pass-meta {
        font-family: 'IBM Plex Mono', ui-monospace, monospace;
        font-size: 11px;
        color: var(--cv-success);
    }
    .cv-banner-fail {
        background: var(--cv-surface);
        border: 1px solid #E4C8C8;
        border-left: 3px solid var(--cv-error);
        border-radius: 2px;
        padding: 0.75rem 0.95rem;
        margin-bottom: 0.85rem;
    }
    .cv-banner-fail-title {
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--cv-error);
        margin-bottom: 0.3rem;
    }
    .cv-banner-fail-body {
        font-size: 13px;
        color: #5C2E2E;
        line-height: 1.45;
        margin-bottom: 0.4rem;
    }
    .cv-banner-fail-meta {
        font-family: 'IBM Plex Mono', ui-monospace, monospace;
        font-size: 11px;
        color: var(--cv-error);
        background: #F7F0F0;
        padding: 0.25rem 0.5rem;
        border-radius: 2px;
        display: inline-block;
    }

    /* ── Prediction ── */
    .cv-pred-caption {
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--cv-text-sec);
        margin-bottom: 0.25rem;
    }
    .cv-pred-heading {
        font-size: 28px;
        font-weight: 700;
        letter-spacing: -0.02em;
        line-height: 1.15;
        margin-bottom: 0.15rem;
    }
    .cv-pred-cancer { color: var(--cv-error); }
    .cv-pred-divert { color: var(--cv-primary); }
    .cv-conf-block { margin-bottom: 0.85rem; }
    .cv-conf-label {
        font-size: 12px;
        font-weight: 500;
        color: var(--cv-text-sec);
    }
    .cv-conf-value {
        font-size: 34px;
        font-weight: 700;
        color: var(--cv-text);
        font-variant-numeric: tabular-nums;
        font-family: 'IBM Plex Mono', ui-monospace, monospace;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }

    /* Thin probability meters */
    .cv-meter-row {
        display: flex;
        justify-content: space-between;
        font-size: 12px;
        margin-bottom: 0.2rem;
    }
    .cv-meter-name { font-weight: 600; color: var(--cv-text); }
    .cv-meter-val {
        font-weight: 600;
        color: var(--cv-text);
        font-family: 'IBM Plex Mono', ui-monospace, monospace;
        font-size: 11px;
    }
    .cv-meter-track {
        height: 3px;
        background: var(--cv-surface-muted);
        border-radius: 1px;
        overflow: hidden;
        margin-bottom: 0.55rem;
    }
    .cv-meter-fill-cancer { height: 100%; background: var(--cv-error); }
    .cv-meter-fill-divert { height: 100%; background: var(--cv-primary); }

    /* Info rows */
    .cv-info-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.4rem 0;
        border-bottom: 1px solid var(--cv-surface-muted);
        font-size: 13px;
    }
    .cv-info-row:last-child { border-bottom: none; }
    .cv-info-key { color: var(--cv-text-sec); font-weight: 400; }
    .cv-info-val { font-weight: 600; color: var(--cv-text); }

    /* Explainability steps */
    .cv-exp-step {
        display: flex;
        align-items: baseline;
        gap: 0.5rem;
        padding: 0.35rem 0;
        font-size: 13px;
        border-bottom: 1px solid var(--cv-surface-muted);
    }
    .cv-exp-step:last-child { border-bottom: none; }
    .cv-exp-num {
        font-family: 'IBM Plex Mono', ui-monospace, monospace;
        font-size: 11px;
        font-weight: 600;
        color: var(--cv-sage);
        min-width: 1.4rem;
    }
    .cv-exp-note {
        font-size: 12px;
        color: var(--cv-text-sec);
        line-height: 1.45;
        margin-top: 0.55rem;
        padding-top: 0.5rem;
        border-top: 1px solid var(--cv-surface-muted);
    }

    /* Empty state */
    .cv-empty {
        background: var(--cv-surface);
        border: 1px dashed var(--cv-border);
        border-radius: 3px;
        padding: 2.4rem 1.5rem;
        text-align: center;
        margin-bottom: 0.9rem;
    }
    .cv-empty-title {
        font-size: 14px;
        font-weight: 700;
        color: var(--cv-primary);
        margin-bottom: 0.3rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .cv-empty-desc {
        font-size: 13px;
        color: var(--cv-text-sec);
        max-width: 420px;
        margin: 0 auto;
        line-height: 1.45;
    }

    /* History log cards */
    .cv-hist-card {
        background: var(--cv-surface);
        border: 1px solid var(--cv-border);
        border-radius: 3px;
        padding: 0.75rem 0.95rem;
        margin-bottom: 0.55rem;
    }
    .cv-hist-id {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--cv-primary);
        margin-bottom: 0.35rem;
    }
    .cv-hist-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.25rem 1rem;
        font-size: 12px;
    }
    .cv-hist-k { color: var(--cv-text-sec); }
    .cv-hist-v { font-weight: 600; color: var(--cv-text); }

    /* Recommended badge */
    .cv-rec {
        display: inline-block;
        font-size: 9px;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--cv-primary);
        background: var(--cv-primary-light);
        padding: 0.1rem 0.35rem;
        border-radius: 2px;
        margin-left: 0.35rem;
        vertical-align: middle;
    }

    /* Buttons */
    button[kind="primary"],
    button[data-testid="stBaseButton-primary"] {
        background-color: var(--cv-primary) !important;
        color: #FFFFFF !important;
        border: 1px solid var(--cv-primary) !important;
        border-radius: 3px !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        min-height: 2.35rem !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
        transition: background-color 0.12s ease, border-color 0.12s ease !important;
    }
    button[kind="primary"] *,
    button[data-testid="stBaseButton-primary"] * {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    button[kind="primary"]:hover,
    button[data-testid="stBaseButton-primary"]:hover {
        background-color: var(--cv-primary-dark) !important;
        border-color: var(--cv-primary-dark) !important;
    }

    button[kind="secondary"],
    button[data-testid="stBaseButton-secondary"] {
        background-color: var(--cv-surface) !important;
        color: var(--cv-text) !important;
        border: 1px solid #BAC5BE !important;
        border-radius: 3px !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        min-height: 2.35rem !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
        transition: background-color 0.12s ease, border-color 0.12s ease, color 0.12s ease !important;
    }
    button[kind="secondary"] *,
    button[data-testid="stBaseButton-secondary"] * {
        color: var(--cv-text) !important;
        font-weight: 600 !important;
    }
    button[kind="secondary"]:hover,
    button[data-testid="stBaseButton-secondary"]:hover {
        background-color: #F1F6F3 !important;
        border-color: var(--cv-primary) !important;
        color: var(--cv-primary) !important;
    }
    button[kind="secondary"]:hover *,
    button[data-testid="stBaseButton-secondary"]:hover * {
        color: var(--cv-primary) !important;
    }

    div[data-testid="stDownloadButton"] button {
        background-color: var(--cv-surface) !important;
        color: var(--cv-primary) !important;
        border: 1.5px solid var(--cv-primary) !important;
        border-radius: 3px !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        min-height: 2.35rem !important;
        padding: 0.45rem 1rem !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
    }
    div[data-testid="stDownloadButton"] button:hover {
        background-color: var(--cv-primary-light) !important;
        color: var(--cv-primary-dark) !important;
    }

    /* Disclaimer */
    .cv-disclaimer {
        font-size: 12px;
        color: var(--cv-text-sec);
        border-top: 1px solid var(--cv-border);
        padding-top: 0.85rem;
        margin-top: 1.6rem;
        line-height: 1.5;
        text-align: left;
    }
    .cv-disclaimer b { color: var(--cv-text); font-weight: 600; }

    /* Uploader */
    div[data-testid="stFileUploader"] { padding: 0; }
    div[data-testid="stFileUploader"] label {
        color: var(--cv-text) !important;
        font-size: 13px !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em !important;
        margin-bottom: 0.35rem !important;
    }
    div[data-testid="stFileUploader"] label p {
        color: var(--cv-text) !important;
        font-size: 13px !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em !important;
    }
    div[data-testid="stFileUploader"] section {
        border-radius: 4px !important;
        border: 1px dashed #BAC5BE !important;
        background: var(--cv-surface) !important;
        padding: 0.75rem 0.95rem !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02) !important;
        transition: border-color 0.15s ease, background-color 0.15s ease !important;
    }
    div[data-testid="stFileUploader"] section:hover {
        border-color: var(--cv-primary) !important;
        background: #FAFBF9 !important;
    }
    div[data-testid="stFileUploader"] section [data-testid="stFileUploaderDropzoneInstructions"],
    div[data-testid="stFileUploader"] section span,
    div[data-testid="stFileUploader"] section p {
        color: #2D3D35 !important;
        font-size: 13px !important;
        font-weight: 500 !important;
    }
    div[data-testid="stFileUploader"] section small {
        color: var(--cv-text-sec) !important;
        font-size: 11px !important;
        font-weight: 500 !important;
        letter-spacing: 0.01em !important;
    }
    div[data-testid="stFileUploader"] section button,
    div[data-testid="stFileUploader"] section button[kind="secondary"] {
        background-color: #FFFFFF !important;
        color: var(--cv-text) !important;
        border: 1px solid #BAC5BE !important;
        border-radius: 3px !important;
        font-weight: 600 !important;
        font-size: 12px !important;
        padding: 0.35rem 0.85rem !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
        transition: all 0.12s ease !important;
    }
    div[data-testid="stFileUploader"] section button:hover {
        background-color: #EFF5F1 !important;
        border-color: var(--cv-primary) !important;
        color: var(--cv-primary) !important;
    }
    div[data-testid="stFileUploader"] section button * {
        color: inherit !important;
        font-weight: 600 !important;
    }
    div[data-testid="stFileUploaderFile"] {
        background: #F1F6F3 !important;
        border: 1px solid #CAD6CF !important;
        border-radius: 3px !important;
        color: var(--cv-text) !important;
    }
    div[data-testid="stFileUploaderFile"] * {
        color: var(--cv-text) !important;
    }

    /* Selectbox */
    div[data-testid="stSelectbox"] div[data-baseweb="select"] {
        background-color: var(--cv-surface) !important;
        border: 1px solid #BAC5BE !important;
        border-radius: 4px !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"]:hover {
        border-color: var(--cv-primary) !important;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] * {
        color: var(--cv-text) !important;
        font-size: 13px !important;
        font-weight: 600 !important;
    }
    div[data-testid="stSelectbox"] svg {
        fill: var(--cv-primary) !important;
        color: var(--cv-primary) !important;
        opacity: 1 !important;
    }

    /* Slider */
    div[data-testid="stSlider"] label {
        font-size: 12px !important;
        font-weight: 600 !important;
        color: var(--cv-text) !important;
    }
    div[data-testid="stSlider"] div[data-testid="stThumbValue"] {
        color: var(--cv-text) !important;
        font-weight: 600 !important;
        font-size: 11px !important;
    }

    /* Caption & Privacy Text */
    div[data-testid="stCaptionContainer"] p,
    .stCaption,
    small.stCaption {
        color: var(--cv-text-sec) !important;
        font-size: 11.5px !important;
        line-height: 1.45 !important;
        font-weight: 500 !important;
    }

    /* Tooltip / Help Icon */
    [data-testid="stTooltipIcon"] svg,
    [data-testid="stTooltipHoverTarget"] svg,
    button[aria-label="Help"] svg {
        color: var(--cv-text-sec) !important;
        fill: var(--cv-text-sec) !important;
        opacity: 0.85 !important;
    }
    [data-testid="stTooltipIcon"]:hover svg,
    [data-testid="stTooltipHoverTarget"]:hover svg {
        color: var(--cv-primary) !important;
        fill: var(--cv-primary) !important;
        opacity: 1 !important;
    }

    /* Select / metrics polish */
    div[data-testid="stMetricValue"] {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        color: var(--cv-text) !important;
    }
    div[data-testid="stMetricLabel"] p {
        color: var(--cv-text-sec) !important;
        font-weight: 500 !important;
    }

    /* Hide Streamlit chrome noise */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)

# ── Constants & Paths ─────────────────────────────────────────────────────────
MODEL_PATHS = {
    "resnet50": PROJECT_ROOT / "models" / "resnet50_best.pth",
    "densenet121": PROJECT_ROOT / "models" / "densenet121_best.pth",
    "efficientnet_b0": PROJECT_ROOT / "models" / "efficientnet_b0_best.pth",
}

MODEL_DISPLAY = {
    "resnet50": "ResNet-50",
    "densenet121": "DenseNet-121",
    "efficientnet_b0": "EfficientNet-B0",
}

MODEL_OPTIONS = [
    "ResNet-50",
    "DenseNet-121",
    "EfficientNet-B0",
]

DISPLAY_TO_KEY = {
    "ResNet-50": "resnet50",
    "DenseNet-121": "densenet121",
    "EfficientNet-B0": "efficientnet_b0",
}

# Existing study benchmark values (from results/model_comparison.csv) — display only
BENCHMARK_ROWS = [
    {
        "Model": "ResNet-50",
        "Parameters": "23.5M",
        "Test Accuracy": "100.0%",
        "Macro-F1": "1.0000",
        "ROC-AUC": "1.0000",
        "Diverticula Recall": "100.0%",
    },
    {
        "Model": "DenseNet-121",
        "Parameters": "6.9M",
        "Test Accuracy": "96.15%",
        "Macro-F1": "0.9328",
        "ROC-AUC": "1.0000",
        "Diverticula Recall": "80.0%",
    },
    {
        "Model": "EfficientNet-B0",
        "Parameters": "4.0M",
        "Test Accuracy": "96.15%",
        "Macro-F1": "0.9328",
        "ROC-AUC": "0.9333",
        "Diverticula Recall": "80.0%",
    },
]

TEST_DIR = PROJECT_ROOT / "data" / "processed" / "test"
VALIDATOR_PROFILE_PATH = PROJECT_ROOT / "models" / "endoscopy_validator_profile.json"

CLASS_NAMES = ["colon_diverticula", "colorectal_cancer"]
CLASS_LABELS = {
    "colon_diverticula": "Colon Diverticula",
    "colorectal_cancer": "Colorectal Cancer",
}
NUM_CLASSES = 2
IMG_SIZE = 224

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Deterministic evaluation pipeline (unchanged)
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

NAV_WORKSPACE = ["Analysis", "History"]
NAV_RESEARCH = ["Model Performance", "Explainability"]
NAV_SYSTEM = ["System Status", "About"]
ALL_NAV = NAV_WORKSPACE + NAV_RESEARCH + NAV_SYSTEM


# ── Model Loaders (Cached) — architectures unchanged ──────────────────────────
@st.cache_resource(show_spinner="Initializing clinical backbone...")
def load_backbone(model_key: str):
    """Load a trained checkpoint. Architecture heads match training pipeline."""
    checkpoint_path = MODEL_PATHS.get(model_key)
    if checkpoint_path is None or not checkpoint_path.exists():
        return None, f"Model checkpoint not found for: {model_key}"

    try:
        if model_key == "resnet50":
            model = resnet50(weights=None)
            model.fc = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(model.fc.in_features, NUM_CLASSES),
            )
        elif model_key == "densenet121":
            model = densenet121(weights=None)
            model.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(model.classifier.in_features, NUM_CLASSES),
            )
        elif model_key == "efficientnet_b0":
            model = efficientnet_b0(weights=None)
            in_f = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(in_f, NUM_CLASSES),
            )
        else:
            return None, f"Unknown model key: {model_key}"

        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        return model, None
    except Exception as e:
        return None, f"Failed to initialize model: {str(e)}"


def load_model(checkpoint_path: Path):
    """
    Backward-compatible ResNet-50 loader used by tests/test_app_inference.py.
    Does not alter architecture, weights, or preprocessing.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        return None, f"Model checkpoint not found at: {checkpoint_path}"
    try:
        model = resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(model.fc.in_features, NUM_CLASSES),
        )
        state_dict = torch.load(path, map_location=DEVICE)
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


def get_gradcam_target_layer(model: nn.Module, model_key: str):
    """
    Select the final convolutional block for Grad-CAM.
    - ResNet-50: layer4[-1] (unchanged)
    - DenseNet-121: features.denseblock4 (final dense block before norm5/relu inplace)
    - EfficientNet-B0: features[-1] (final conv head)
    """
    if model_key == "resnet50":
        return model.layer4[-1]
    if model_key == "densenet121":
        return model.features.denseblock4
    if model_key == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(f"No Grad-CAM target for: {model_key}")


def gradcam_layer_label(model_key: str) -> str:
    if model_key == "resnet50":
        return "layer4[-1]"
    if model_key == "densenet121":
        return "features.denseblock4"
    if model_key == "efficientnet_b0":
        return "features[-1]"
    return "—"


# ── Grad-CAM Extractor (algorithm unchanged) ──────────────────────────────────
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
            self.activations = output.clone() if isinstance(output, torch.Tensor) else output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].clone() if isinstance(grad_out[0], torch.Tensor) else grad_out[0]

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self.hook_handles.extend([h1, h2])

    def compute(self, input_tensor: torch.Tensor, target_class: int = None):
        try:
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

            weights = torch.mean(grads, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * acts, dim=1, keepdim=True)
            cam = torch.relu(cam)

            cam_np = cam.squeeze().cpu().numpy()
            cam_min, cam_max = cam_np.min(), cam_np.max()
            if cam_max - cam_min > 1e-8:
                cam_norm = (cam_np - cam_min) / (cam_max - cam_min)
            else:
                cam_norm = np.zeros_like(cam_np)

            return (
                cam_norm,
                pred_class,
                confidence,
                probs[0].detach().cpu().numpy(),
                logits[0].detach().cpu().numpy(),
            )
        finally:
            self.remove()

    def remove(self):
        for h in self.hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self.hook_handles = []


def generate_overlay(pil_crop: Image.Image, cam_heatmap: np.ndarray, alpha: float = 0.45):
    img_np = np.array(pil_crop).astype(np.float32) / 255.0

    cam_pil = Image.fromarray((cam_heatmap * 255).astype(np.uint8)).resize(
        (IMG_SIZE, IMG_SIZE), resample=Image.BICUBIC
    )
    cam_resized = np.array(cam_pil).astype(np.float32) / 255.0

    try:
        import matplotlib
        colormap = matplotlib.colormaps["jet"]
    except Exception:
        colormap = cm.get_cmap("jet")
    heatmap_rgb = colormap(cam_resized)[:, :, :3]

    overlay = (1.0 - alpha) * img_np + alpha * heatmap_rgb
    overlay = np.clip(overlay, 0.0, 1.0)

    heatmap_uint8 = (heatmap_rgb * 255).astype(np.uint8)
    overlay_uint8 = (overlay * 255).astype(np.uint8)

    return Image.fromarray(heatmap_uint8), Image.fromarray(overlay_uint8)


# ── Session State ─────────────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "nav_page": "Analysis",
        "current_image": None,
        "current_source_label": "",
        "analysis_result": None,
        "history": [],
        "viewer_tab": "OVERLAY",
        "heatmap_alpha": 0.45,
        "uploader_key": 0,
        "selected_model": "ResNet-50",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


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


def render_disclaimer():
    st.markdown("""
    <div class="cv-disclaimer">
        <b>Research prototype.</b> ColonVision AI is an AI-assisted research analysis system for academic
        demonstration and is <b>not</b> a clinical diagnostic device. It must not be used as a substitute
        for licensed medical professionals, endoscopic assessment, or histopathology.
        The current dataset is limited in size. Grad-CAM is an interpretability aid and does not
        establish clinical diagnosis or lesion localization.
    </div>
    """, unsafe_allow_html=True)


def render_header(model_display: str):
    st.markdown(f"""
    <div class="cv-header">
        <div class="cv-header-left">
            <div class="cv-header-brand">ColonVision AI</div>
            <div class="cv-header-sub">Clinical Research Intelligence</div>
        </div>
        <div class="cv-header-right">
            <span class="cv-header-tag">Clinical Research System</span>
            <span class="cv-status-pill">
                <span class="cv-dot cv-dot-ok"></span>
                Operational
            </span>
            <span class="cv-model-pill">{model_display}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Main Application ──────────────────────────────────────────────────────────
def main():
    init_session_state()

    model_key = DISPLAY_TO_KEY[st.session_state.selected_model]
    model, model_err = load_backbone(model_key)
    if model_err:
        st.error(f"Backbone Error: {model_err}")
        st.stop()

    validator, val_err = load_validator(VALIDATOR_PROFILE_PATH)
    if val_err:
        st.error(f"Validator Error: {val_err}")
        st.stop()

    # ── LEFT SIDEBAR ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div class="cv-side-brand">
            <div class="cv-side-name">ColonVision</div>
            <div class="cv-side-ai">AI</div>
            <div class="cv-side-tag">Clinical Research Intelligence</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("New Analysis", use_container_width=True, type="primary", key="btn_side_new_analysis"):
            reset_analysis_state()
            st.session_state.nav_page = "Analysis"
            st.rerun()

        st.markdown('<div class="cv-nav-group">Workspace</div>', unsafe_allow_html=True)
        for item in NAV_WORKSPACE:
            label = f"• {item}"
            if st.button(label, use_container_width=True, key=f"nav_ws_{item}",
                         type="primary" if st.session_state.nav_page == item else "secondary"):
                st.session_state.nav_page = item
                st.rerun()

        st.markdown('<div class="cv-nav-group">Research</div>', unsafe_allow_html=True)
        for item in NAV_RESEARCH:
            label = f"• {item}"
            if st.button(label, use_container_width=True, key=f"nav_rs_{item}",
                         type="primary" if st.session_state.nav_page == item else "secondary"):
                st.session_state.nav_page = item
                st.rerun()

        st.markdown('<div class="cv-nav-group">System</div>', unsafe_allow_html=True)
        for item in NAV_SYSTEM:
            label = f"• {item}"
            if st.button(label, use_container_width=True, key=f"nav_sy_{item}",
                         type="primary" if st.session_state.nav_page == item else "secondary"):
                st.session_state.nav_page = item
                st.rerun()

        st.markdown("""
        <div class="cv-side-status">
            <div class="cv-side-status-val">
                <span class="cv-dot cv-dot-ok"></span>
                SYSTEM OPERATIONAL
            </div>
        </div>
        """, unsafe_allow_html=True)

    render_header(st.session_state.selected_model)

    # ── PAGE: ANALYSIS ────────────────────────────────────────────────────────
    if st.session_state.nav_page == "Analysis":
        # Top: Image Analysis controls
        st.markdown("""
        <div class="cv-card">
            <div class="cv-section-label">Image Analysis</div>
            <div class="cv-section-desc">
                Upload a colonoscopy image for domain verification, model inference and explainability analysis.
            </div>
        """, unsafe_allow_html=True)

        up_col, act_col = st.columns([2.2, 1.8], gap="medium")

        with up_col:
            uploaded_file = st.file_uploader(
                "Upload Image",
                type=["jpg", "jpeg", "png"],
                key=f"uploader_{st.session_state.uploader_key}",
                help="Upload a colonoscopy image in JPG, JPEG, or PNG format.",
            )
            if uploaded_file is not None:
                new_label = f"Uploaded: {uploaded_file.name}"
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
            st.caption("Images are processed locally for this research prototype and are not intentionally stored by the interface.")

        with act_col:
            st.markdown('<div class="cv-micro" style="margin-bottom:0.4rem;">Actions</div>', unsafe_allow_html=True)
            b1, b2 = st.columns(2)
            with b1:
                if st.button("Sample Cancer", use_container_width=True, key="btn_sample_cancer"):
                    load_clinical_sample("cancer")
                    st.rerun()
            with b2:
                if st.button("Sample Diverticula", use_container_width=True, key="btn_sample_divert"):
                    load_clinical_sample("divert")
                    st.rerun()

            b3, b4 = st.columns(2)
            with b3:
                run_requested = False
                if st.session_state.current_image is not None and st.session_state.analysis_result is None:
                    run_requested = st.button("Analyze", type="primary", use_container_width=True, key="btn_analyze_now")
                elif st.session_state.current_image is not None and st.session_state.analysis_result is not None:
                    if st.button("Analyze", type="primary", use_container_width=True, key="btn_analyze_rerun"):
                        st.session_state.analysis_result = None
                        st.rerun()
            with b4:
                if st.button("Clear", use_container_width=True, key="btn_clear_image"):
                    reset_analysis_state()
                    st.rerun()

            # Model selector
            st.markdown('<div class="cv-micro" style="margin:0.65rem 0 0.35rem 0;">Selected Model</div>', unsafe_allow_html=True)
            model_choice = st.selectbox(
                "Model",
                options=MODEL_OPTIONS,
                index=MODEL_OPTIONS.index(st.session_state.selected_model),
                format_func=lambda x: f"{x}  · Recommended" if x == "ResNet-50" else x,
                key="model_select_box",
                label_visibility="collapsed",
                help="ResNet-50 is marked Recommended because it is the best-performing model in the existing study benchmark.",
            )
            if model_choice != st.session_state.selected_model:
                st.session_state.selected_model = model_choice
                st.session_state.analysis_result = None
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        # Empty state — no image
        if st.session_state.current_image is None:
            col_v, col_r = st.columns([1.45, 1], gap="medium")
            with col_v:
                st.markdown("""
                <div class="cv-card">
                    <div class="cv-card-title"><span>Image Viewer</span></div>
                    <div class="cv-viewer">
                        <div class="cv-viewer-empty">
                            <div class="cv-viewer-empty-title">No Image Selected</div>
                            <div class="cv-viewer-empty-desc">Upload a colonoscopy image to begin analysis.</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col_r:
                st.markdown(f"""
                <div class="cv-card">
                    <div class="cv-card-title"><span>Domain Verification</span></div>
                    <div class="cv-info-row">
                        <span class="cv-info-key">Status</span>
                        <span class="cv-info-val">Awaiting input</span>
                    </div>
                </div>
                <div class="cv-card">
                    <div class="cv-card-title"><span>Selected Model</span></div>
                    <div class="cv-info-row">
                        <span class="cv-info-key">Architecture</span>
                        <span class="cv-info-val">{st.session_state.selected_model}</span>
                    </div>
                    <div class="cv-info-row">
                        <span class="cv-info-key">Model status</span>
                        <span class="cv-info-val" style="color:#2A6B52;">Ready</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            render_disclaimer()
            return

        current_img = st.session_state.current_image

        # Preview — image loaded, not yet analyzed
        if st.session_state.analysis_result is None and not run_requested:
            col_v, col_r = st.columns([1.45, 1], gap="medium")
            with col_v:
                st.markdown("""
                <div class="cv-card">
                    <div class="cv-card-title">
                        <span>Image Viewer</span>
                        <span style="color:#2A6B52;font-weight:600;font-size:10px;letter-spacing:0.06em;">READY</span>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown('<div class="cv-viewer">', unsafe_allow_html=True)
                st.image(crop_transform(current_img), use_container_width=True)
                st.markdown("</div></div>", unsafe_allow_html=True)
                st.caption(f"Input · {st.session_state.current_source_label}")

            with col_r:
                st.markdown(f"""
                <div class="cv-card">
                    <div class="cv-card-title"><span>Domain Verification</span></div>
                    <div class="cv-info-row">
                        <span class="cv-info-key">Status</span>
                        <span class="cv-info-val">Pending analysis</span>
                    </div>
                </div>
                <div class="cv-card">
                    <div class="cv-card-title"><span>Selected Model</span></div>
                    <div class="cv-info-row">
                        <span class="cv-info-key">Architecture</span>
                        <span class="cv-info-val">{st.session_state.selected_model}</span>
                    </div>
                    <div class="cv-info-row">
                        <span class="cv-info-key">Model status</span>
                        <span class="cv-info-val" style="color:#2A6B52;">Ready</span>
                    </div>
                </div>
                <div class="cv-card">
                    <div class="cv-card-title"><span>Analysis Summary</span></div>
                    <div class="cv-section-desc" style="margin-bottom:0;">
                        Click <b>Analyze</b> to run domain verification and model inference.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            render_disclaimer()
            return

        # Execute analysis
        if st.session_state.analysis_result is None and run_requested:
            with st.spinner("Executing domain verification..."):
                val_res = validator.validate_image(current_img)

            if not val_res["is_endoscopy"]:
                st.session_state.analysis_result = {
                    "is_endoscopy": False,
                    "domain_score": val_res["domain_score"],
                    "threshold": val_res["threshold"],
                    "model_name": st.session_state.selected_model,
                    "model_key": model_key,
                }
            else:
                with st.spinner(f"Computing {st.session_state.selected_model} prediction and Grad-CAM..."):
                    input_tensor = eval_transform(current_img).unsqueeze(0).to(DEVICE)
                    cropped_original = crop_transform(current_img)

                    target_layer = get_gradcam_target_layer(model, model_key)
                    extractor = GradCAMExtractor(model, target_layer)
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
                        "model_name": st.session_state.selected_model,
                        "model_key": model_key,
                    }
                    st.session_state.analysis_result = res_payload

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
                        "model_name": st.session_state.selected_model,
                        "model_key": model_key,
                        "threshold": float(val_res["threshold"]),
                    })

        res = st.session_state.analysis_result

        # Domain rejection
        if not res["is_endoscopy"]:
            st.markdown(f"""
            <div class="cv-banner-fail">
                <div class="cv-banner-fail-title">Input Rejected</div>
                <div class="cv-banner-fail-body">
                    Analysis was stopped because the submitted image does not meet the validated endoscopy-domain criteria.
                    Non-endoscopic imagery is rejected to prevent spurious classifications.
                </div>
                <div class="cv-banner-fail-meta">
                    Domain Similarity Score: <b>{res['domain_score']:.4f}</b>
                    &nbsp;|&nbsp; Calibrated Threshold: <b>&ge; {res['threshold']:.4f}</b>
                    &nbsp;|&nbsp; Status: <b>REJECTED</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_v, col_r = st.columns([1.45, 1], gap="medium")
            with col_v:
                st.markdown("""
                <div class="cv-card">
                    <div class="cv-card-title"><span>Image Viewer</span></div>
                """, unsafe_allow_html=True)
                st.markdown('<div class="cv-viewer">', unsafe_allow_html=True)
                st.image(crop_transform(current_img), use_container_width=True)
                st.markdown("</div></div>", unsafe_allow_html=True)
            with col_r:
                st.markdown(f"""
                <div class="cv-card">
                    <div class="cv-card-title"><span>Domain Verification</span></div>
                    <div class="cv-info-row">
                        <span class="cv-info-key">Status</span>
                        <span class="cv-info-val" style="color:#8F3D3D;">INPUT REJECTED</span>
                    </div>
                    <div class="cv-info-row">
                        <span class="cv-info-key">Score</span>
                        <span class="cv-info-val">{res['domain_score']:.4f}</span>
                    </div>
                    <div class="cv-info-row">
                        <span class="cv-info-key">Threshold</span>
                        <span class="cv-info-val">&ge; {res['threshold']:.4f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Choose Another Image", type="primary", key="btn_choose_another", use_container_width=True):
                    reset_analysis_state()
                    st.rerun()
            render_disclaimer()
            return

        # Successful analysis — three-zone center + right
        cropped_orig = res["cropped_original"]
        cam_map = res["cam_map"]
        pred_class = res["pred_class"]
        conf = res["confidence"]
        probs = res["probs"]
        used_model = res.get("model_name", st.session_state.selected_model)
        used_key = res.get("model_key", model_key)

        prob_divert = float(probs[0])
        prob_cancer = float(probs[1])

        heatmap_img, overlay_img = generate_overlay(
            cropped_orig, cam_map, alpha=st.session_state.heatmap_alpha
        )

        st.markdown(f"""
        <div class="cv-banner-pass">
            <div class="cv-banner-pass-title">Verified Endoscopy Input</div>
            <div class="cv-banner-pass-meta">
                Domain Score: <b>{res['domain_score']:.4f}</b>
                &nbsp;|&nbsp; Threshold: <b>&ge; {res['threshold']:.4f}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_viewer, col_ctx = st.columns([1.45, 1], gap="medium")

        with col_viewer:
            st.markdown("""
            <div class="cv-card">
                <div class="cv-card-title">
                    <span>Image Viewer</span>
                    <span style="color:#1A4D3E;font-weight:600;font-size:10px;letter-spacing:0.06em;">GRAD-CAM</span>
                </div>
            """, unsafe_allow_html=True)

            vt1, vt2, vt3 = st.columns(3)
            with vt1:
                if st.button(
                    "ORIGINAL",
                    use_container_width=True,
                    type="primary" if st.session_state.viewer_tab == "ORIGINAL" else "secondary",
                    key="tab_original",
                ):
                    st.session_state.viewer_tab = "ORIGINAL"
                    st.rerun()
            with vt2:
                if st.button(
                    "GRAD-CAM",
                    use_container_width=True,
                    type="primary" if st.session_state.viewer_tab == "GRAD-CAM" else "secondary",
                    key="tab_gradcam",
                ):
                    st.session_state.viewer_tab = "GRAD-CAM"
                    st.rerun()
            with vt3:
                if st.button(
                    "OVERLAY",
                    use_container_width=True,
                    type="primary" if st.session_state.viewer_tab == "OVERLAY" else "secondary",
                    key="tab_overlay",
                ):
                    st.session_state.viewer_tab = "OVERLAY"
                    st.rerun()

            st.markdown('<div class="cv-viewer">', unsafe_allow_html=True)
            if st.session_state.viewer_tab == "ORIGINAL":
                st.image(cropped_orig, use_container_width=True)
            elif st.session_state.viewer_tab == "GRAD-CAM":
                st.image(heatmap_img, use_container_width=True)
            else:
                st.image(overlay_img, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            sl_col, res_col = st.columns([3, 1])
            with sl_col:
                new_alpha = st.slider(
                    "Opacity",
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
                st.markdown("<div style='height:1.45rem;'></div>", unsafe_allow_html=True)
                if st.button("Reset", use_container_width=True, key="btn_reset_alpha"):
                    st.session_state.heatmap_alpha = 0.45
                    st.rerun()

            st.markdown(f"""
            <div class="cv-exp-note">
                Target layer: <code>{gradcam_layer_label(used_key)}</code>.
                Highlighted regions represent image areas that contributed most strongly to the model prediction.
                Grad-CAM is an interpretability aid and does not establish clinical diagnosis.
            </div>
            </div>
            """, unsafe_allow_html=True)

        with col_ctx:
            # Domain Verification
            st.markdown(f"""
            <div class="cv-card">
                <div class="cv-card-title"><span>Domain Verification</span></div>
                <div class="cv-info-row">
                    <span class="cv-info-key">Status</span>
                    <span class="cv-info-val" style="color:#2A6B52;">VERIFIED ENDOSCOPY INPUT</span>
                </div>
                <div class="cv-info-row">
                    <span class="cv-info-key">Score</span>
                    <span class="cv-info-val">{res['domain_score']:.4f}</span>
                </div>
                <div class="cv-info-row">
                    <span class="cv-info-key">Threshold</span>
                    <span class="cv-info-val">&ge; {res['threshold']:.4f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Prediction
            name_cls = "cv-pred-cancer" if pred_class == "colorectal_cancer" else "cv-pred-divert"
            formatted_name = CLASS_LABELS[pred_class]

            st.markdown(f"""
            <div class="cv-card">
                <div class="cv-card-title"><span>Prediction</span></div>
                <div class="cv-pred-caption">Model Prediction</div>
                <div class="cv-pred-heading {name_cls}">{formatted_name}</div>
                <div class="cv-conf-block">
                    <div class="cv-conf-label">Confidence</div>
                    <div class="cv-conf-value">{conf * 100:.2f}%</div>
                </div>
                <div class="cv-meter-row">
                    <span class="cv-meter-name">Colorectal Cancer</span>
                    <span class="cv-meter-val">{prob_cancer * 100:.2f}%</span>
                </div>
                <div class="cv-meter-track">
                    <div class="cv-meter-fill-cancer" style="width:{max(1, int(prob_cancer * 100))}%;"></div>
                </div>
                <div class="cv-meter-row">
                    <span class="cv-meter-name">Colon Diverticula</span>
                    <span class="cv-meter-val">{prob_divert * 100:.2f}%</span>
                </div>
                <div class="cv-meter-track">
                    <div class="cv-meter-fill-divert" style="width:{max(1, int(prob_divert * 100))}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Model information
            st.markdown(f"""
            <div class="cv-card">
                <div class="cv-card-title"><span>Model Information</span></div>
                <div class="cv-info-row">
                    <span class="cv-info-key">Selected Model</span>
                    <span class="cv-info-val">{used_model}</span>
                </div>
                <div class="cv-info-row">
                    <span class="cv-info-key">Model status</span>
                    <span class="cv-info-val" style="color:#2A6B52;">Ready</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Explainability
            st.markdown("""
            <div class="cv-card">
                <div class="cv-card-title"><span>Explainability</span></div>
                <div class="cv-micro" style="margin-bottom:0.45rem;">Grad-CAM</div>
                <div class="cv-exp-step"><span class="cv-exp-num">01</span><span>ORIGINAL</span></div>
                <div class="cv-exp-step"><span class="cv-exp-num">02</span><span>GRAD-CAM</span></div>
                <div class="cv-exp-step"><span class="cv-exp-num">03</span><span>OVERLAY</span></div>
                <div class="cv-exp-note">
                    Highlighted regions represent image areas that contributed most strongly to the model prediction.
                    Grad-CAM is an interpretability aid and does not establish clinical diagnosis.
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Analysis summary
            st.markdown(f"""
            <div class="cv-card">
                <div class="cv-card-title"><span>Analysis Summary</span></div>
                <div class="cv-info-row">
                    <span class="cv-info-key">Input</span>
                    <span class="cv-info-val">Endoscopy image</span>
                </div>
                <div class="cv-info-row">
                    <span class="cv-info-key">Domain</span>
                    <span class="cv-info-val" style="color:#2A6B52;">Verified</span>
                </div>
                <div class="cv-info-row">
                    <span class="cv-info-key">Model</span>
                    <span class="cv-info-val">{used_model}</span>
                </div>
                <div class="cv-info-row">
                    <span class="cv-info-key">Prediction</span>
                    <span class="cv-info-val">{formatted_name}</span>
                </div>
                <div class="cv-info-row">
                    <span class="cv-info-key">Confidence</span>
                    <span class="cv-info-val">{conf * 100:.2f}%</span>
                </div>
                <div class="cv-info-row">
                    <span class="cv-info-key">Explainability</span>
                    <span class="cv-info-val">Grad-CAM available</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

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
                    model_name=used_model,
                    threshold=float(res.get("threshold", 0.6800)),
                )
                ts_str = datetime.now().strftime("%Y-%m-%d_%H%M")
                st.download_button(
                    label="Download Analysis Report",
                    data=report_bytes,
                    file_name=f"ColonVision_AI_Analysis_Report_{ts_str}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="btn_download_analysis_report",
                )
            except Exception:
                st.error("Unable to generate the report. Please try again.")
                if st.button("Try Again", key="btn_try_again_report", use_container_width=True):
                    st.rerun()

    # ── PAGE: HISTORY ─────────────────────────────────────────────────────────
    elif st.session_state.nav_page == "History":
        st.markdown("""
        <div class="cv-card">
            <div class="cv-card-title">
                <span>Analysis History</span>
                <span style="font-weight:600;color:var(--cv-text-sec);font-size:11px;">Clinical Analysis Log · Session Only</span>
            </div>
        """, unsafe_allow_html=True)

        if not st.session_state.history:
            st.markdown("""
            <div class="cv-empty">
                <div class="cv-empty-title">No Analyses Recorded</div>
                <div class="cv-empty-desc">
                    Return to Analysis to run domain verification and model inference.
                    History is stored in-session only and is not persisted remotely.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for rec in reversed(st.session_state.history):
                st.markdown(f"""
                <div class="cv-hist-card">
                    <div class="cv-hist-id">Analysis {rec['id']:02d}</div>
                    <div class="cv-hist-grid">
                        <div><span class="cv-hist-k">Prediction</span><br><span class="cv-hist-v">{rec['prediction']}</span></div>
                        <div><span class="cv-hist-k">Confidence</span><br><span class="cv-hist-v">{rec['confidence']}</span></div>
                        <div><span class="cv-hist-k">Model</span><br><span class="cv-hist-v">{rec.get('model_name', 'ResNet-50')}</span></div>
                        <div><span class="cv-hist-k">Time</span><br><span class="cv-hist-v">{rec['timestamp']}</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            h_c1, h_c2 = st.columns([3, 1])
            with h_c1:
                hist_ids = [rec["id"] for rec in st.session_state.history]
                selected_id = st.selectbox(
                    "Select analysis record",
                    options=hist_ids,
                    format_func=lambda x: (
                        f"Analysis {x:02d} · {st.session_state.history[x - 1]['prediction']} · "
                        f"{st.session_state.history[x - 1]['confidence']} · "
                        f"{st.session_state.history[x - 1]['timestamp']}"
                    ),
                    key="hist_select_box",
                )
            with h_c2:
                st.markdown("<div style='height:1.65rem;'></div>", unsafe_allow_html=True)
                if st.button("Clear History", use_container_width=True, key="btn_clear_history"):
                    st.session_state.history = []
                    st.session_state.analysis_result = None
                    st.rerun()

            if selected_id is not None and len(st.session_state.history) >= selected_id:
                sel_rec = st.session_state.history[selected_id - 1]
                st.markdown("---")
                det_c1, det_c2 = st.columns([1, 1.25])
                with det_c1:
                    st.markdown(f"**Analysis {sel_rec['id']:02d}**")
                    st.markdown(f"Prediction: `{sel_rec['prediction']}`")
                    st.markdown(f"Confidence: `{sel_rec['confidence']}`")
                    st.markdown(f"Model: `{sel_rec.get('model_name', 'ResNet-50')}`")
                    st.markdown(f"Time: `{sel_rec['timestamp']}`")
                    st.markdown(f"Domain Score: `{sel_rec['domain_score']:.4f}`")

                    if st.button(
                        "Load into Workspace",
                        type="primary",
                        key=f"btn_reopen_{sel_rec['id']}",
                        use_container_width=True,
                    ):
                        st.session_state.current_image = sel_rec["cropped_original"]
                        st.session_state.current_source_label = (
                            f"Historical Record #{sel_rec['id']}: {sel_rec['input']}"
                        )
                        st.session_state.selected_model = sel_rec.get("model_name", "ResNet-50")
                        st.session_state.analysis_result = {
                            "is_endoscopy": True,
                            "domain_score": sel_rec["domain_score"],
                            "threshold": float(sel_rec.get("threshold", 0.6800)),
                            "pred_idx": 1 if "Cancer" in sel_rec["prediction"] else 0,
                            "pred_class": (
                                "colorectal_cancer"
                                if "Cancer" in sel_rec["prediction"]
                                else "colon_diverticula"
                            ),
                            "confidence": float(sel_rec["confidence"].replace("%", "")) / 100.0,
                            "probs": sel_rec["probs"],
                            "logits": np.array([0.0, 0.0]),
                            "cam_map": sel_rec["cam_map"],
                            "cropped_original": sel_rec["cropped_original"],
                            "model_name": sel_rec.get("model_name", "ResNet-50"),
                            "model_key": sel_rec.get("model_key", "resnet50"),
                        }
                        st.session_state.nav_page = "Analysis"
                        st.rerun()

                    try:
                        hist_h, hist_o = generate_overlay(
                            sel_rec["cropped_original"], sel_rec["cam_map"], alpha=0.45
                        )
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
                            model_name=sel_rec.get("model_name", "ResNet-50"),
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
                    st.markdown('<div class="cv-viewer">', unsafe_allow_html=True)
                    _, hist_ov = generate_overlay(
                        sel_rec["cropped_original"], sel_rec["cam_map"], alpha=0.45
                    )
                    st.image(hist_ov, caption="Grad-CAM Overlay", use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── PAGE: MODEL PERFORMANCE ───────────────────────────────────────────────
    elif st.session_state.nav_page == "Model Performance":
        st.markdown("""
        <div class="cv-card">
            <div class="cv-card-title">
                <span>Model Performance</span>
                <span style="font-weight:600;color:var(--cv-text-sec);font-size:11px;">Independent Test Set (26 Images)</span>
            </div>
            <div class="cv-section-desc">
                Comparison of transfer-learning backbones trained under identical hyperparameters
                on the GastroVision benchmark. Values are from the existing study results — not regenerated.
            </div>
        """, unsafe_allow_html=True)

        st.table(BENCHMARK_ROWS)

        st.markdown("""
        <div class="cv-exp-note">
            <b>Note:</b> ResNet-50 achieved zero false-positives and zero false-negatives on the held-out
            test split (21 colorectal cancer, 5 colon diverticula). Because the test set is limited in size,
            5-fold cross-validation was additionally executed across the 142 development images.
            ResNet-50 is marked Recommended solely based on this existing benchmark.
        </div>
        </div>
        """, unsafe_allow_html=True)

    # ── PAGE: EXPLAINABILITY ──────────────────────────────────────────────────
    elif st.session_state.nav_page == "Explainability":
        st.markdown("""
        <div class="cv-card">
            <div class="cv-card-title">
                <span>Explainability</span>
                <span style="font-weight:500;color:#1A4D3E;">GRAD-CAM</span>
            </div>
            <div style="font-size:14px;color:#1C1F1D;line-height:1.55;margin-bottom:0.75rem;">
                <b>Gradient-weighted Class Activation Mapping (Grad-CAM)</b> provides visual transparency
                into neural network decisions without retraining or architecture modification.
            </div>
            <div class="cv-micro" style="margin-bottom:0.45rem;">Visualization Sequence</div>
            <div class="cv-exp-step"><span class="cv-exp-num">01</span><span>ORIGINAL — preprocessed input crop</span></div>
            <div class="cv-exp-step"><span class="cv-exp-num">02</span><span>GRAD-CAM — class activation heatmap</span></div>
            <div class="cv-exp-step"><span class="cv-exp-num">03</span><span>OVERLAY — interpretability visualization</span></div>
            <div class="cv-exp-note">
                Highlighted regions represent image areas that contributed most strongly to the model prediction.
                Grad-CAM is an interpretability aid and does not establish clinical diagnosis.
                Target layer for ResNet-50 remains <code>layer4[-1]</code> as in the existing study.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── PAGE: SYSTEM STATUS ───────────────────────────────────────────────────
    elif st.session_state.nav_page == "System Status":
        st.markdown("""
        <div class="cv-card">
            <div class="cv-card-title">
                <span>System Status</span>
                <span style="color:#2A6B52;font-weight:600;">● OPERATIONAL</span>
            </div>
        """, unsafe_allow_html=True)

        sys_cols = st.columns(3)
        with sys_cols[0]:
            st.metric("Active Model", st.session_state.selected_model)
            st.metric("Input Resolution", "224 × 224 px")
        with sys_cols[1]:
            st.metric("Inference Engine", f"PyTorch ({DEVICE.type.upper()})")
            st.metric("Grad-CAM Target", gradcam_layer_label(model_key))
        with sys_cols[2]:
            st.metric("Validator Status", "Active")
            st.metric("Calibrated Threshold", f"τ* = {validator.threshold:.4f}")

        st.markdown("""
        <div class="cv-exp-note">
            <b>Integrity:</b> Checkpoint weights, endoscopy validator profile, and deterministic
            evaluation transforms verified for this research workstation session.
            Available backbones: ResNet-50, DenseNet-121, EfficientNet-B0.
        </div>
        </div>
        """, unsafe_allow_html=True)

    # ── PAGE: ABOUT ───────────────────────────────────────────────────────────
    elif st.session_state.nav_page == "About":
        st.markdown("""
        <div class="cv-card">
            <div class="cv-card-title">
                <span>About</span>
                <span style="font-weight:600;color:var(--cv-text-sec);font-size:11px;">Research Project</span>
            </div>
            <div style="font-size:14px;color:#1C1F1D;line-height:1.55;margin-bottom:0.75rem;">
                <b>ColonVision AI</b> is an engineering research prototype for AI-assisted research analysis
                of colorectal cancer and colon diverticula from colonoscopy images using transfer learning
                and Grad-CAM interpretability visualization.
            </div>
            <div style="font-size:13px;color:var(--cv-text-sec);line-height:1.55;">
                <b>Core components:</b>
                <ul>
                    <li><b>Endoscopy Domain Validator</b> — cosine-similarity feature matching to reject non-endoscopic imagery prior to classification.</li>
                    <li><b>Multi-model Classifier</b> — ResNet-50, DenseNet-121, and EfficientNet-B0 fine-tuned on GastroVision endoscopic frames.</li>
                    <li><b>Grad-CAM Explainability</b> — channel-weighted activation mapping on the final convolutional block of the selected architecture.</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

    render_disclaimer()


if __name__ == "__main__":
    main()
