#!/usr/bin/env python3
"""
Test harness for ColonVision AI Streamlit app inference and Grad-CAM logic.
"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import app components directly
from app.app import load_model, GradCAMExtractor, eval_transform, crop_transform, generate_overlay, CLASS_NAMES

def test_inference_pipeline():
    model_path = PROJECT_ROOT / "models" / "resnet50_best.pth"
    print(f"Testing model loading from: {model_path}")
    model, err = load_model(model_path)
    assert err is None, f"Model loading failed: {err}"
    assert model is not None, "Model object is None"
    print("[PASS] Model successfully loaded.")

    # Test Case 1: Colorectal Cancer
    cancer_img_path = PROJECT_ROOT / "data" / "processed" / "test" / "colorectal_cancer" / "0c7db439-d40d-468f-9cc7-97a2a526dd7c.jpg"
    if not cancer_img_path.exists():
        cancer_img_path = PROJECT_ROOT / "app" / "samples" / "sample_colorectal_cancer.jpg"
    assert cancer_img_path.exists(), f"Cancer test image not found: {cancer_img_path}"
    img_cancer = Image.open(cancer_img_path).convert("RGB")

    t_tensor = eval_transform(img_cancer).unsqueeze(0)
    extractor = GradCAMExtractor(model, model.layer4[-1])
    cam_map, pred_idx, conf, probs, logits = extractor.compute(t_tensor)
    extractor.remove()

    pred_name = CLASS_NAMES[pred_idx]
    print(f"\n[Test 1 - Cancer Image: {cancer_img_path.name}]")
    print(f"  Predicted: {pred_name} (Class {pred_idx})")
    print(f"  Confidence: {conf:.4f} ({conf*100:.2f}%)")
    print(f"  P(Diverticula): {probs[0]:.4f} | P(Cancer): {probs[1]:.4f}")
    assert pred_name == "colorectal_cancer", f"Expected colorectal_cancer, got {pred_name}"
    assert conf > 0.5, f"Confidence too low: {conf}"
    assert cam_map.shape == (7, 7), f"Unexpected CAM shape: {cam_map.shape}"
    assert cam_map.min() >= 0.0 and cam_map.max() <= 1.0, "CAM out of range"

    # Test overlay generation
    cropped = crop_transform(img_cancer)
    hm_img, ov_img = generate_overlay(cropped, cam_map, alpha=0.45)
    assert hm_img.size == (224, 224), f"Heatmap size error: {hm_img.size}"
    assert ov_img.size == (224, 224), f"Overlay size error: {ov_img.size}"
    print("  [PASS] Grad-CAM heatmap and overlay generated successfully.")

    # Test Case 2: Colon Diverticula
    divert_img_path = PROJECT_ROOT / "data" / "processed" / "test" / "colon_diverticula" / "04675a9b-4858-439c-9199-94426a1e76a5.jpg"
    if not divert_img_path.exists():
        divert_img_path = PROJECT_ROOT / "app" / "samples" / "sample_colon_diverticula.jpg"
    assert divert_img_path.exists(), f"Diverticula test image not found: {divert_img_path}"
    img_divert = Image.open(divert_img_path).convert("RGB")

    t_tensor = eval_transform(img_divert).unsqueeze(0)
    extractor = GradCAMExtractor(model, model.layer4[-1])
    cam_map, pred_idx, conf, probs, logits = extractor.compute(t_tensor)
    extractor.remove()

    pred_name = CLASS_NAMES[pred_idx]
    print(f"\n[Test 2 - Diverticula Image: {divert_img_path.name}]")
    print(f"  Predicted: {pred_name} (Class {pred_idx})")
    print(f"  Confidence: {conf:.4f} ({conf*100:.2f}%)")
    print(f"  P(Diverticula): {probs[0]:.4f} | P(Cancer): {probs[1]:.4f}")
    assert pred_name == "colon_diverticula", f"Expected colon_diverticula, got {pred_name}"
    assert conf > 0.5, f"Confidence too low: {conf}"
    assert cam_map.shape == (7, 7), f"Unexpected CAM shape: {cam_map.shape}"

    # Test overlay generation
    cropped = crop_transform(img_divert)
    hm_img, ov_img = generate_overlay(cropped, cam_map, alpha=0.45)
    assert hm_img.size == (224, 224), f"Heatmap size error: {hm_img.size}"
    assert ov_img.size == (224, 224), f"Overlay size error: {ov_img.size}"
    print("  [PASS] Grad-CAM heatmap and overlay generated successfully.")

    print("\n" + "="*60)
    print("ALL APP INFERENCE & GRAD-CAM TESTS PASSED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    test_inference_pipeline()
