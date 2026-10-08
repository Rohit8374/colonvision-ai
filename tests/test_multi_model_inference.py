#!/usr/bin/env python3
"""
Automated Multi-Model Inference Verification Suite
===================================================
Tests all three trained clinical backbones on CPU:
1. ResNet-50        (models/resnet50_best.pth)
2. DenseNet-121     (models/densenet121_best.pth)
3. EfficientNet-B0  (models/efficientnet_b0_best.pth)

Validates:
- Successful checkpoint loading on CPU (map_location='cpu')
- Evaluation mode and zero gradients
- Inference output logits and softmax probabilities
- Accurate classification on benchmark test images
"""

import sys
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CLASS_NAMES = ["colon_diverticula", "colorectal_cancer"]

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_backbone(name: str, checkpoint_path: Path):
    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    if name == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.fc.in_features, 2))
        m.load_state_dict(state_dict)
    elif name == "densenet121":
        m = models.densenet121(weights=None)
        m.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.classifier.in_features, 2))
        m.load_state_dict(state_dict)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.classifier[1].in_features, 2))
        m.load_state_dict(state_dict)
    else:
        raise ValueError(f"Unknown backbone: {name}")

    m.eval()
    return m


def run_multi_model_tests():
    print("=" * 70)
    print("COLONVISION AI — Multi-Model Inference Test Suite (CPU)")
    print("=" * 70)

    models_dir = PROJECT_ROOT / "models"
    test_dir = PROJECT_ROOT / "data" / "processed" / "test"

    cancer_img_path = test_dir / "colorectal_cancer" / "0c7db439-d40d-468f-9cc7-97a2a526dd7c.jpg"
    if not cancer_img_path.exists():
        cancer_img_path = PROJECT_ROOT / "app" / "samples" / "sample_colorectal_cancer.jpg"
    divert_img_path = test_dir / "colon_diverticula" / "04675a9b-4858-439c-9199-94426a1e76a5.jpg"
    if not divert_img_path.exists():
        divert_img_path = PROJECT_ROOT / "app" / "samples" / "sample_colon_diverticula.jpg"

    assert cancer_img_path.exists(), f"Cancer image not found: {cancer_img_path}"
    assert divert_img_path.exists(), f"Diverticula image not found: {divert_img_path}"

    with Image.open(cancer_img_path) as im:
        t_cancer = eval_transform(im.convert("RGB")).unsqueeze(0)
    with Image.open(divert_img_path) as im:
        t_divert = eval_transform(im.convert("RGB")).unsqueeze(0)

    configs = [
        ("resnet50", models_dir / "resnet50_best.pth", "ResNet-50"),
        ("densenet121", models_dir / "densenet121_best.pth", "DenseNet-121"),
        ("efficientnet_b0", models_dir / "efficientnet_b0_best.pth", "EfficientNet-B0"),
    ]

    for model_key, ckpt_path, display_name in configs:
        print(f"\n--- Testing Model: {display_name} ({ckpt_path.name}) ---")
        model = load_backbone(model_key, ckpt_path)
        print(f"  [PASS] Checkpoint loaded successfully on CPU.")

        # Test Cancer sample
        with torch.no_grad():
            out_cancer = model(t_cancer)
            probs_cancer = torch.softmax(out_cancer, dim=1)[0].cpu().numpy()
            pred_cancer = int(probs_cancer.argmax())

        print(f"  [Cancer Test Image]")
        print(f"    Predicted: {CLASS_NAMES[pred_cancer]} (Class {pred_cancer})")
        print(f"    P(Diverticula): {probs_cancer[0]:.4f} | P(Cancer): {probs_cancer[1]:.4f}")
        assert CLASS_NAMES[pred_cancer] == "colorectal_cancer", f"Expected colorectal_cancer, got {CLASS_NAMES[pred_cancer]}"
        assert probs_cancer[1] > 0.5, f"Low cancer confidence: {probs_cancer[1]}"
        print(f"    [PASS] Correct classification with {probs_cancer[1]*100:.2f}% confidence.")

        # Test Diverticula sample
        with torch.no_grad():
            out_divert = model(t_divert)
            probs_divert = torch.softmax(out_divert, dim=1)[0].cpu().numpy()
            pred_divert = int(probs_divert.argmax())

        print(f"  [Diverticula Test Image]")
        print(f"    Predicted: {CLASS_NAMES[pred_divert]} (Class {pred_divert})")
        print(f"    P(Diverticula): {probs_divert[0]:.4f} | P(Cancer): {probs_divert[1]:.4f}")
        assert CLASS_NAMES[pred_divert] == "colon_diverticula", f"Expected colon_diverticula, got {CLASS_NAMES[pred_divert]}"
        assert probs_divert[0] > 0.5, f"Low diverticula confidence: {probs_divert[0]}"
        print(f"    [PASS] Correct classification with {probs_divert[0]*100:.2f}% confidence.")

    print("\n" + "=" * 70)
    print("ALL THREE MODELS (ResNet-50, DenseNet-121, EfficientNet-B0) PASSED!")
    print("=" * 70)


def run_model_switching_gradcam_regression_test():
    """
    Regression test for dynamic model switching with Grad-CAM extraction.
    Tests sequential transitions: ResNet-50 -> DenseNet-121 -> EfficientNet-B0 -> ResNet-50.
    Ensures that backward hooks and autograd graph modifications do not cause runtime
    errors or persist across model transitions.
    """
    import numpy as np
    from app.app import get_gradcam_target_layer, GradCAMExtractor

    print("\n" + "=" * 70)
    print("REGRESSION TEST: Sequential Model Switching with Grad-CAM")
    print("=" * 70)

    models_dir = PROJECT_ROOT / "models"
    cancer_img_path = PROJECT_ROOT / "app" / "samples" / "sample_colorectal_cancer.jpg"
    with Image.open(cancer_img_path) as im:
        t_cancer = eval_transform(im.convert("RGB")).unsqueeze(0)

    switch_sequence = [
        ("resnet50", models_dir / "resnet50_best.pth", "ResNet-50"),
        ("densenet121", models_dir / "densenet121_best.pth", "DenseNet-121"),
        ("efficientnet_b0", models_dir / "efficientnet_b0_best.pth", "EfficientNet-B0"),
        ("resnet50", models_dir / "resnet50_best.pth", "ResNet-50 (Switchback)"),
    ]

    for step_idx, (model_key, ckpt_path, display_name) in enumerate(switch_sequence, 1):
        print(f"\n[Step {step_idx}/4] Switching to: {display_name}")
        model = load_backbone(model_key, ckpt_path)
        target_layer = get_gradcam_target_layer(model, model_key)
        assert target_layer is not None, f"Target layer is None for {model_key}"

        extractor = GradCAMExtractor(model, target_layer)
        cam, pred_class, conf, probs, logits = extractor.compute(t_cancer)

        assert isinstance(cam, np.ndarray), f"Expected numpy ndarray for CAM, got {type(cam)}"
        assert not np.isnan(cam).any(), "Grad-CAM contains NaNs"
        assert cam.ndim == 2, f"Expected 2D heatmap, got shape {cam.shape}"
        assert cam.min() >= 0.0 and cam.max() <= 1.0, f"Grad-CAM not normalized in [0, 1]: min={cam.min()}, max={cam.max()}"
        assert CLASS_NAMES[pred_class] == "colorectal_cancer", f"Incorrect prediction: {CLASS_NAMES[pred_class]}"
        assert len(extractor.hook_handles) == 0, "GradCAMExtractor hooks were not cleaned up!"
        print(f"  [PASS] Grad-CAM generated: shape={cam.shape}, pred={CLASS_NAMES[pred_class]} ({conf*100:.2f}%), hooks removed.")

    print("\n" + "=" * 70)
    print("MODEL SWITCHING & GRAD-CAM REGRESSION TEST PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    run_multi_model_tests()
    run_model_switching_gradcam_regression_test()
