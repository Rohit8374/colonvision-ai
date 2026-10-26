"""
ColonVision AI — Unified Neural Inference & Saliency Engine
=============================================================
Encapsulates PyTorch model loading, evaluation transforms, and
architecture-specific Grad-CAM extraction without Streamlit dependencies.
Used by FastAPI backend and test harnesses.
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, densenet121, efficientnet_b0

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

CLASS_NAMES = ["colon_diverticula", "colorectal_cancer"]
NUM_CLASSES = 2
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATHS = {
    "resnet50": MODELS_DIR / "resnet50_best.pth",
    "densenet121": MODELS_DIR / "densenet121_best.pth",
    "efficientnet_b0": MODELS_DIR / "efficientnet_b0_best.pth",
}

MODEL_DISPLAY = {
    "resnet50": "ResNet-50",
    "densenet121": "DenseNet-121",
    "efficientnet_b0": "EfficientNet-B0",
}

MODEL_CONFIGS = {
    "densenet121": {
        "name": "DenseNet-121",
        "params": "8.0M",
        "f1": "0.9328",
        "description": "Dense connectivity with gradient preservation. Optimal baseline.",
    },
    "resnet50": {
        "name": "ResNet-50",
        "params": "25.6M",
        "f1": "1.0000",
        "description": "Standard Residual Deep Network. Baseline clinical benchmark.",
    },
    "efficientnet_b0": {
        "name": "EfficientNet-B0",
        "params": "5.3M",
        "f1": "0.9328",
        "description": "Compound scaling optimized for edge & mobile frame telemetry.",
    },
}

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

crop_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
])


def load_backbone(model_key: str) -> Tuple[Optional[nn.Module], Optional[str]]:
    """Loads a fine-tuned clinical model checkpoint matching trained head architectures."""
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
        return None, f"Failed to initialize model {model_key}: {str(e)}"


def get_gradcam_target_layer(model: nn.Module, model_key: str) -> nn.Module:
    """Target final conv block for spatial gradient backprop."""
    if model_key == "resnet50":
        return model.layer4[-1]
    if model_key == "densenet121":
        return model.features.denseblock4
    if model_key == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(f"No Grad-CAM target defined for: {model_key}")


def gradcam_layer_label(model_key: str) -> str:
    if model_key == "resnet50":
        return "layer4[-1]"
    if model_key == "densenet121":
        return "features.denseblock4"
    if model_key == "efficientnet_b0":
        return "features[-1]"
    return "—"


class GradCAMExtractor:
    """Hooks into target convolutional layer to compute class activation maps."""
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

    def compute(self, input_tensor: torch.Tensor, target_class: Optional[int] = None):
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
            target_score.backward()

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
        self.activations = None
        self.gradients = None
        try:
            self.model.zero_grad(set_to_none=True)
        except Exception:
            pass
        self.model = None
