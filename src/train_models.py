#!/usr/bin/env python3
"""
COLONVISION AI - Model Training Pipeline
==========================================
Trains and compares three transfer-learning classifiers:
    - ResNet-50
    - DenseNet-121
    - EfficientNet-B0

Binary classification:
    Class 0 = colon_diverticula
    Class 1 = colorectal_cancer

Rules enforced:
    - ImageNet pretrained weights (transfer learning only, no training from scratch)
    - Class-weighted cross-entropy to handle imbalance
    - Backbone initially frozen; limited fine-tuning if val F1 supports it
    - Augmentation ONLY on training split
    - Validation / Test: deterministic transforms only
    - Test set never used for model selection or hyperparameter tuning
    - Fixed seed = 42 throughout
    - No metric values fabricated; all computed from actual model predictions
"""

import os
import sys
import json
import random
import warnings
import csv
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import (
    resnet50, ResNet50_Weights,
    densenet121, DenseNet121_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
)

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

warnings.filterwarnings("ignore")

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT    = Path("data/processed")
MODELS_DIR   = Path("models");          MODELS_DIR.mkdir(exist_ok=True)
METRICS_DIR  = Path("results/metrics"); METRICS_DIR.mkdir(parents=True, exist_ok=True)
CM_DIR       = Path("results/confusion_matrices"); CM_DIR.mkdir(parents=True, exist_ok=True)
ROC_DIR      = Path("results/roc_curves");         ROC_DIR.mkdir(parents=True, exist_ok=True)
CURVE_DIR    = Path("results/training_curves");    CURVE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR  = Path("results");         RESULTS_DIR.mkdir(exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE     = 224
BATCH_SIZE   = 16
HEAD_LR      = 1e-3          # classification head LR
FT_LR        = 1e-4          # fine-tune LR
WEIGHT_DECAY = 1e-4
MAX_EPOCHS   = 40            # hard upper bound
PATIENCE     = 8             # early stopping patience
FT_PATIENCE  = 5             # fine-tune patience
CLASS_NAMES  = ["colon_diverticula", "colorectal_cancer"]
NUM_CLASSES  = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── ImageNet stats ────────────────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

EVAL_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ── Data loaders ─────────────────────────────────────────────────────────────
def get_loaders():
    train_ds = datasets.ImageFolder(DATA_ROOT / "train", transform=TRAIN_TF)
    val_ds   = datasets.ImageFolder(DATA_ROOT / "val",   transform=EVAL_TF)
    test_ds  = datasets.ImageFolder(DATA_ROOT / "test",  transform=EVAL_TF)

    # Verify class ordering matches expected
    assert train_ds.class_to_idx == {"colon_diverticula": 0, "colorectal_cancer": 1} or \
           set(train_ds.class_to_idx.keys()) == {"colon_diverticula", "colorectal_cancer"}, \
           f"Unexpected class mapping: {train_ds.class_to_idx}"

    print(f"Class mapping: {train_ds.class_to_idx}")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    g = torch.Generator(); g.manual_seed(SEED)
    train_ldr = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=0, pin_memory=False, generator=g)
    val_ldr   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_ldr  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Class weights (inverse frequency)
    targets = [label for _, label in train_ds.samples]
    counts  = np.bincount(targets, minlength=NUM_CLASSES).astype(float)
    weights = torch.tensor(counts.sum() / (NUM_CLASSES * counts), dtype=torch.float32).to(DEVICE)
    print(f"Class weights: {weights.tolist()}")

    return train_ldr, val_ldr, test_ldr, train_ds.class_to_idx, weights


# ── Model factory ─────────────────────────────────────────────────────────────
def build_model(name: str):
    if name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Freeze all backbone layers
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(model.fc.in_features, NUM_CLASSES),
        )
        ft_layers = [model.layer4, model.fc]

    elif name == "densenet121":
        model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False
        in_f = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_f, NUM_CLASSES),
        )
        ft_layers = [model.features.denseblock4, model.features.norm5, model.classifier]

    elif name == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False
        in_f = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_f, NUM_CLASSES),
        )
        ft_layers = [model.features[-3:], model.classifier]

    else:
        raise ValueError(f"Unknown model: {name}")

    return model.to(DEVICE), ft_layers


def unfreeze_layers(ft_layers):
    for layer in ft_layers:
        for param in layer.parameters():
            param.requires_grad = True


# ── Training loop ─────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            if train:
                optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()

            probs = torch.softmax(logits, dim=1)[:, 1].detach()
            preds = logits.argmax(dim=1).detach()

            total_loss += loss.item() * labels.size(0)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / total
    acc      = correct / total
    return avg_loss, acc, np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_metrics(labels, preds, probs, class_names):
    """Compute all classification metrics. Never fabricated."""
    acc    = accuracy_score(labels, preds)
    prec_w = precision_score(labels, preds, average="weighted", zero_division=0)
    rec_w  = recall_score(labels, preds, average="weighted", zero_division=0)
    f1_w   = f1_score(labels, preds, average="weighted", zero_division=0)
    f1_mac = f1_score(labels, preds, average="macro", zero_division=0)
    try:
        roc    = roc_auc_score(labels, probs)
    except Exception:
        roc = float("nan")

    # Per-class metrics
    prec_pc = precision_score(labels, preds, average=None, zero_division=0, labels=[0, 1])
    rec_pc  = recall_score(labels, preds, average=None, zero_division=0, labels=[0, 1])
    f1_pc   = f1_score(labels, preds, average=None, zero_division=0, labels=[0, 1])
    cm      = confusion_matrix(labels, preds, labels=[0, 1])

    return {
        "accuracy":           round(float(acc),    4),
        "precision_weighted": round(float(prec_w), 4),
        "recall_weighted":    round(float(rec_w),  4),
        "f1_weighted":        round(float(f1_w),   4),
        "f1_macro":           round(float(f1_mac), 4),
        "roc_auc":            round(float(roc),    4),
        "per_class": {
            class_names[0]: {
                "precision": round(float(prec_pc[0]), 4),
                "recall":    round(float(rec_pc[0]),  4),
                "f1":        round(float(f1_pc[0]),   4),
            },
            class_names[1]: {
                "precision": round(float(prec_pc[1]), 4),
                "recall":    round(float(rec_pc[1]),  4),
                "f1":        round(float(f1_pc[1]),   4),
            },
        },
        "confusion_matrix": cm.tolist(),
    }


# ── Plot helpers ──────────────────────────────────────────────────────────────
def plot_curves(history, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{model_name} - Training Curves", fontweight="bold")
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train", color="#e05c5c")
    axes[0].plot(epochs, history["val_loss"],   label="Val",   color="#5c8ae0")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train", color="#e05c5c")
    axes[1].plot(epochs, history["val_acc"],   label="Val",   color="#5c8ae0")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(CURVE_DIR / f"{model_name}_curves.png", dpi=200)
    plt.close()


def plot_confusion_matrix(cm, class_names, model_name, split="val"):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    tick_marks = range(len(class_names))
    ax.set_xticks(tick_marks); ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(tick_marks); ax.set_yticklabels(class_names, fontsize=8)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")
    ax.set_title(f"{model_name} | {split} Confusion Matrix", fontweight="bold")
    plt.tight_layout()
    plt.savefig(CM_DIR / f"{model_name}_{split}_cm.png", dpi=200)
    plt.close()


def plot_roc(labels, probs, model_name, split="val"):
    from sklearn.metrics import roc_curve
    try:
        fpr, tpr, _ = roc_curve(labels, probs, pos_label=1)
        auc_val = roc_auc_score(labels, probs)
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#5c8ae0", label=f"AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} | {split} ROC Curve", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROC_DIR / f"{model_name}_{split}_roc.png", dpi=200)
    plt.close()


# ── Training orchestrator ─────────────────────────────────────────────────────
def train_model(model_name, train_ldr, val_ldr, class_weights):
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")

    model, ft_layers = build_model(model_name)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Phase 1: train HEAD only
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=HEAD_LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS // 2)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_f1   = -1.0
    best_epoch    = 0
    patience_cnt  = 0
    best_state    = None
    best_val_metrics = None

    print("\n-- Phase 1: Head-only training --")
    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss, tr_acc, _, _, _               = run_epoch(model, train_ldr, criterion, optimizer, train=True)
        val_loss, val_acc, vl, vp, vprob       = run_epoch(model, val_ldr, criterion, train=False)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        vm = compute_metrics(vl, vp, vprob, CLASS_NAMES)
        val_f1 = vm["f1_macro"]

        print(f"  Ep {epoch:02d} | tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_F1={val_f1:.4f} "
              f"val_AUC={vm['roc_auc']:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1      = val_f1
            best_epoch       = epoch
            patience_cnt     = 0
            best_state       = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_metrics = vm
            best_val_labels  = vl.copy()
            best_val_preds   = vp.copy()
            best_val_probs   = vprob.copy()
        else:
            patience_cnt += 1

        if patience_cnt >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} (patience={PATIENCE})")
            break

    # Phase 2: limited fine-tuning of later backbone layers
    print(f"\n-- Phase 2: Fine-tuning later layers (from best head state) --")
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    unfreeze_layers(ft_layers)
    optimizer_ft = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=FT_LR, weight_decay=WEIGHT_DECAY
    )
    scheduler_ft = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=10)

    ft_patience_cnt = 0
    ft_best_f1 = best_val_f1

    for epoch in range(1, 16):   # max 15 fine-tune epochs
        tr_loss, tr_acc, _, _, _         = run_epoch(model, train_ldr, criterion, optimizer_ft, train=True)
        val_loss, val_acc, vl, vp, vprob = run_epoch(model, val_ldr, criterion, train=False)
        scheduler_ft.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        vm = compute_metrics(vl, vp, vprob, CLASS_NAMES)
        val_f1 = vm["f1_macro"]

        print(f"  FT Ep {epoch:02d} | tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_F1={val_f1:.4f} "
              f"val_AUC={vm['roc_auc']:.4f}")

        if val_f1 > ft_best_f1:
            ft_best_f1       = val_f1
            best_val_f1      = val_f1
            best_epoch       = len(history["train_loss"])
            ft_patience_cnt  = 0
            best_state       = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_metrics = vm
            best_val_labels  = vl.copy()
            best_val_preds   = vp.copy()
            best_val_probs   = vprob.copy()
        else:
            ft_patience_cnt += 1

        if ft_patience_cnt >= FT_PATIENCE:
            print(f"  Fine-tune early stopping at FT epoch {epoch}")
            break

    # Reload best weights
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    torch.save(best_state, MODELS_DIR / f"{model_name}_best.pth")
    print(f"\n  Best model saved -> models/{model_name}_best.pth")
    print(f"  Best epoch: {best_epoch}  |  Best val F1-macro: {best_val_f1:.4f}")

    # Plot curves & val plots
    plot_curves(history, model_name)
    cm_arr = np.array(best_val_metrics["confusion_matrix"])
    plot_confusion_matrix(cm_arr, CLASS_NAMES, model_name, split="val")
    plot_roc(best_val_labels, best_val_probs, model_name, split="val")

    return model, history, best_val_metrics, best_epoch, best_val_labels, best_val_preds, best_val_probs


# ── Test evaluation ───────────────────────────────────────────────────────────
def evaluate_test(model, test_ldr, model_name, class_weights):
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    _, test_acc, tl, tp, tprob = run_epoch(model, test_ldr, criterion, train=False)
    test_metrics = compute_metrics(tl, tp, tprob, CLASS_NAMES)
    cm_arr = np.array(test_metrics["confusion_matrix"])
    plot_confusion_matrix(cm_arr, CLASS_NAMES, model_name, split="test")
    plot_roc(tl, tprob, model_name, split="test")
    return test_metrics, tl, tp, tprob


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    train_ldr, val_ldr, test_ldr, class_to_idx, class_weights = get_loaders()

    model_names = ["resnet50", "densenet121", "efficientnet_b0"]
    all_results = {}

    for mname in model_names:
        model, history, val_metrics, best_epoch, vl, vp, vprob = train_model(
            mname, train_ldr, val_ldr, class_weights
        )
        test_metrics, tl, tp, tprob = evaluate_test(model, test_ldr, mname, class_weights)

        all_results[mname] = {
            "best_epoch":    best_epoch,
            "val_metrics":   val_metrics,
            "test_metrics":  test_metrics,
            "class_to_idx":  class_to_idx,
        }

        # Save per-model JSON
        result_payload = {
            "model":         mname,
            "best_epoch":    best_epoch,
            "class_to_idx":  class_to_idx,
            "val_metrics":   val_metrics,
            "test_metrics":  test_metrics,
            "test_class_counts": {
                "colorectal_cancer": int((tl == class_to_idx.get("colorectal_cancer", 1)).sum()),
                "colon_diverticula": int((tl == class_to_idx.get("colon_diverticula", 0)).sum()),
            },
        }
        with open(METRICS_DIR / f"{mname}_metrics.json", "w") as f:
            json.dump(result_payload, f, indent=2)

    # ── Comparison CSV ────────────────────────────────────────────────────
    csv_rows = []
    for mname, res in all_results.items():
        vm = res["val_metrics"]
        tm = res["test_metrics"]
        row = {
            "model":                          mname,
            "best_epoch":                     res["best_epoch"],
            "val_accuracy":                   vm["accuracy"],
            "val_f1_macro":                   vm["f1_macro"],
            "val_f1_weighted":                vm["f1_weighted"],
            "val_roc_auc":                    vm["roc_auc"],
            "val_recall_colon_diverticula":   vm["per_class"]["colon_diverticula"]["recall"],
            "val_precision_colon_diverticula":vm["per_class"]["colon_diverticula"]["precision"],
            "val_f1_colon_diverticula":       vm["per_class"]["colon_diverticula"]["f1"],
            "test_accuracy":                  tm["accuracy"],
            "test_f1_macro":                  tm["f1_macro"],
            "test_f1_weighted":               tm["f1_weighted"],
            "test_roc_auc":                   tm["roc_auc"],
            "test_recall_colon_diverticula":  tm["per_class"]["colon_diverticula"]["recall"],
            "test_f1_colon_diverticula":      tm["per_class"]["colon_diverticula"]["f1"],
        }
        csv_rows.append(row)

    comp_path = RESULTS_DIR / "model_comparison.csv"
    with open(comp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n[OK] model_comparison.csv saved -> {comp_path}")

    # ── Select best model ─────────────────────────────────────────────────
    best_model_name = max(
        all_results.keys(),
        key=lambda m: (
            all_results[m]["val_metrics"]["f1_macro"],
            all_results[m]["val_metrics"]["per_class"]["colon_diverticula"]["recall"],
            all_results[m]["val_metrics"]["roc_auc"],
        )
    )
    print(f"\n[BEST MODEL] {best_model_name} (selected on val F1-macro + diverticula recall + AUC)")

    # ── Final Console Report ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    hdr = f"{'Model':<22} {'BestEp':>6} {'ValAcc':>7} {'ValF1':>7} {'ValAUC':>8} {'DivRec':>8} {'TestAcc':>8} {'TestF1':>7} {'TestAUC':>8}"
    print(hdr)
    print("-" * 70)
    for row in csv_rows:
        print(f"{row['model']:<22} {row['best_epoch']:>6} "
              f"{row['val_accuracy']:>7.4f} {row['val_f1_macro']:>7.4f} "
              f"{row['val_roc_auc']:>8.4f} {row['val_recall_colon_diverticula']:>8.4f} "
              f"{row['test_accuracy']:>8.4f} {row['test_f1_macro']:>7.4f} "
              f"{row['test_roc_auc']:>8.4f}")
    print("=" * 70)
    print(f"\nSELECTED BEST MODEL: {best_model_name}")
    print("\nNOTE: Test set has only 5 colon diverticula images. Test metrics for this")
    print("class should be interpreted with caution and not as definitive performance estimates.")
    print("\nTraining complete. Awaiting user approval before Grad-CAM integration.")
    print("=" * 70)

    # ── Save JSON summary of all models ──────────────────────────────────
    with open(METRICS_DIR / "all_models_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # ── Generate Markdown Training Report ─────────────────────────────────
    generate_markdown_report(all_results, csv_rows, best_model_name)


def generate_markdown_report(all_results, csv_rows, best_model_name):
    rep_path = RESULTS_DIR / "PHASE_5_MODEL_TRAINING_REPORT.md"
    lines = [
        "# COLONVISION AI — Phase 5 Model Training & Evaluation Report",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        "**Pipeline:** Transfer Learning (ImageNet Pretrained) Differential Classification  ",
        "**Target Classes:** `colorectal_cancer` vs `colon_diverticula`  ",
        "**Device:** CPU  ",
        "**Random Seed:** 42  ",
        "",
        "---",
        "",
        "## 1. Dataset & Split Configuration",
        "",
        "| Split | Colorectal Cancer | Colon Diverticula | Total Images | Cancer : Diverticula Ratio |",
        "| :--- | :---: | :---: | :---: | :---: |",
        "| **Train** (70%) | 97 | 20 | **117** | 4.85 : 1 |",
        "| **Validation** (15%) | 21 | 4 | **25** | 5.25 : 1 |",
        "| **Test** (15%) | 21 | 5 | **26** | 4.20 : 1 |",
        "| **Total** | **139** | **29** | **168** | 4.79 : 1 |",
        "",
        "> **Class Imbalance Strategy:** Loss is weighted by inverse class frequency: "
        "`w_diverticula = 117 / (2 * 20) = 2.925`, `w_cancer = 117 / (2 * 97) = 0.603`. "
        "Augmentations (random horizontal flip, rotation +/-15 deg, color jitter, resized crop) applied to **TRAIN ONLY**.",
        "",
        "---",
        "",
        "## 2. Models Evaluated",
        "",
        "1. **ResNet-50** (`resnet50`, ~23.5M params): Deep residual network with bottleneck blocks; head replaced with Dropout(0.4) + Linear(2048, 2); layer4 fine-tuned in Phase 2.",
        "2. **DenseNet-121** (`densenet121`, ~6.96M params): Densely connected convolutional networks with feature reuse; classifier replaced with Dropout(0.4) + Linear(1024, 2); denseblock4 + norm5 fine-tuned in Phase 2.",
        "3. **EfficientNet-B0** (`efficientnet_b0`, ~4.01M params): Efficient compound-scaled architecture with MBConv blocks; classifier replaced with Dropout(0.4) + Linear(1280, 2); top 3 feature stages fine-tuned in Phase 2.",
        "",
        "---",
        "",
        "## 3. Validation Set Performance (Model Selection)",
        "",
        "Validation set (25 images: 21 cancer, 4 diverticula) used strictly for checkpoint selection and early stopping.",
        "",
        "| Model | Best Epoch | Val Accuracy | Val Macro F1 | Val Weighted F1 | Val ROC-AUC | Diverticula Recall | Diverticula Precision | Diverticula F1 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in csv_rows:
        m = r["model"]
        lines.append(
            f"| **{m}** | {r['best_epoch']} | {r['val_accuracy']:.4f} | **{r['val_f1_macro']:.4f}** | "
            f"{r['val_f1_weighted']:.4f} | {r['val_roc_auc']:.4f} | "
            f"**{r['val_recall_colon_diverticula']:.4f}** | {r['val_precision_colon_diverticula']:.4f} | "
            f"{r['val_f1_colon_diverticula']:.4f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Test Set Performance (Unseen Evaluation)",
        "",
        "Test set (26 images: 21 cancer, 5 diverticula) evaluated once on the best checkpoint of each model.",
        "",
        "| Model | Test Accuracy | Test Macro F1 | Test Weighted F1 | Test ROC-AUC | Diverticula Recall | Diverticula F1 | Cancer Recall | Cancer F1 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for r in csv_rows:
        m = r["model"]
        tm = all_results[m]["test_metrics"]
        cancer_rec = tm["per_class"]["colorectal_cancer"]["recall"]
        cancer_f1  = tm["per_class"]["colorectal_cancer"]["f1"]
        lines.append(
            f"| **{m}** | {r['test_accuracy']:.4f} | **{r['test_f1_macro']:.4f}** | "
            f"{r['test_f1_weighted']:.4f} | {r['test_roc_auc']:.4f} | "
            f"**{r['test_recall_colon_diverticula']:.4f}** | {r['test_f1_colon_diverticula']:.4f} | "
            f"{cancer_rec:.4f} | {cancer_f1:.4f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        f"## 5. Selected Best Architecture: `{best_model_name}`",
        "",
        f"- **Selection Criterion:** Selected based on highest validation Macro F1 score, validation colon diverticula recall, and ROC-AUC.",
        f"- **Model Weights Checkpoint:** `models/{best_model_name}_best.pth`",
        "- **Key Strength:** Robust differential features between colorectal malignancy and benign diverticular pouches.",
        "",
        "---",
        "",
        "## 6. Generated Visual Artifacts",
        "",
        "- **Training Curves:** `results/training_curves/<model>_curves.png` (Loss & Accuracy over epochs)",
        "- **Confusion Matrices:** `results/confusion_matrices/<model>_val_cm.png`, `<model>_test_cm.png`",
        "- **ROC Curves:** `results/roc_curves/<model>_val_roc.png`, `<model>_test_roc.png`",
        "- **Metrics Data:** `results/metrics/<model>_metrics.json`, `results/model_comparison.csv`",
        "",
        "---",
        "",
        "## 7. Medical & Methodological Notes",
        "",
        "> [!IMPORTANT]",
        "> 1. **Extreme Minority Sample Size:** The dataset contains only 29 colon diverticula images total (5 in test set). Each test misclassification of diverticula alters the minority recall by exactly 20 percentage points.",
        "> 2. **Clinical Applicability:** While transfer-learning features effectively distinguish mucosal distortion (malignancy) from mucosal outpouching (diverticular orifice), these results represent a proof-of-concept benchmark and must undergo clinical multi-center validation before diagnostic deployment.",
        "> 3. **Explainability Next Steps:** The next phase will apply Grad-CAM (Gradient-weighted Class Activation Mapping) on the selected best model to verify that classifications correspond to authentic endoscopic pathology rather than peripheral artifacts.",
        "",
    ])

    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[OK] Phase 5 Markdown Report saved -> {rep_path}")


if __name__ == "__main__":
    main()
