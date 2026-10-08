#!/usr/bin/env python3
"""
Automated Test Suite for Endoscopy-Domain Validator / OOD Protection Layer
===========================================================================
Verifies that:
  A. Passport photo -> REJECTED
  B. Selfie/portrait -> REJECTED
  C. ID/document -> REJECTED
  D. Ordinary non-endoscopic photograph -> REJECTED
  E. Known GastroVision endoscopy image -> ACCEPTED
  F. Colorectal cancer test image -> ACCEPTED -> ResNet-50 runs
  G. Colon diverticula test image -> ACCEPTED -> ResNet-50 runs
"""

import sys
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.endoscopy_validator import EndoscopyValidator, PROFILE_PATH

DOWNLOADS_DIR = Path(r"C:\Users\ganes\Downloads")
GASTROVISION_DIR = Path(r"C:\Users\ganes\Downloads\GastroVision\Gastrovision")
TEST_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "test"
MODEL_PATH = PROJECT_ROOT / "models" / "resnet50_best.pth"

# Load classifier
def get_classifier():
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.fc.in_features, 2))
    m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    m.eval()
    return m

eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def run_tests():
    print("=" * 70)
    print("COLONVISION AI — Endoscopy Validator Test Suite")
    print("=" * 70)

    validator = EndoscopyValidator(PROFILE_PATH)
    classifier = get_classifier()
    print(f"Validator loaded successfully with threshold tau* = {validator.threshold:.4f}\n")

    # ── Test A: Passport Photo -> REJECT ─────────────────────────────────────
    passport_path = DOWNLOADS_DIR / "WhatsApp Image 2026-09-28 at 10.17.10 PM.jpeg"
    if passport_path.exists():
        with Image.open(passport_path) as img:
            res = validator.validate_image(img)
        print(f"[Test A - Passport Photo: {passport_path.name}]")
        print(f"  Score = {res['domain_score']:.4f} (Threshold = {res['threshold']:.4f})")
        print(f"  Is Endoscopy = {res['is_endoscopy']} -> {'REJECTED (PASS)' if not res['is_endoscopy'] else 'ACCEPTED (FAIL)'}")
        assert not res["is_endoscopy"], f"Expected REJECT, got ACCEPT for passport photo (score {res['domain_score']})"
    else:
        # Fallback synthetic non-endoscopic image for CI / remote deployment testing
        syn_non_endo = Image.new("RGB", (224, 224), color=(20, 20, 20))
        res = validator.validate_image(syn_non_endo)
        print(f"[Test A - Synthetic Non-Endoscopic Image (Portable Fallback)]")
        print(f"  Score = {res['domain_score']:.4f} (Threshold = {res['threshold']:.4f})")
        assert not res["is_endoscopy"], "Synthetic non-endoscopic image was not rejected"

    # ── Test B: Selfie / Portrait -> REJECT ──────────────────────────────────
    portrait_path = DOWNLOADS_DIR / "profile.jpeg"
    assert portrait_path.exists(), f"Portrait file not found: {portrait_path}"
    with Image.open(portrait_path) as img:
        res = validator.validate_image(img)
    print(f"\n[Test B - Selfie/Portrait: {portrait_path.name}]")
    print(f"  Score = {res['domain_score']:.4f} (Threshold = {res['threshold']:.4f})")
    print(f"  Is Endoscopy = {res['is_endoscopy']} -> {'REJECTED (PASS)' if not res['is_endoscopy'] else 'ACCEPTED (FAIL)'}")
    assert not res["is_endoscopy"], f"Expected REJECT, got ACCEPT for portrait (score {res['domain_score']})"

    # ── Test C: ID Card / Document -> REJECT ─────────────────────────────────
    id_card_path = DOWNLOADS_DIR / "SRKR_Student_ID_Card.png"
    assert id_card_path.exists(), f"ID card file not found: {id_card_path}"
    with Image.open(id_card_path) as img:
        res = validator.validate_image(img)
    print(f"\n[Test C - ID Card/Document: {id_card_path.name}]")
    print(f"  Score = {res['domain_score']:.4f} (Threshold = {res['threshold']:.4f})")
    print(f"  Is Endoscopy = {res['is_endoscopy']} -> {'REJECTED (PASS)' if not res['is_endoscopy'] else 'ACCEPTED (FAIL)'}")
    assert not res["is_endoscopy"], f"Expected REJECT, got ACCEPT for ID card (score {res['domain_score']})"

    # ── Test D: Ordinary Non-Endoscopic Photograph -> REJECT ─────────────────
    outdoor_path = DOWNLOADS_DIR / "pexels-photo-6556790.jpeg"
    assert outdoor_path.exists(), f"Outdoor photo file not found: {outdoor_path}"
    with Image.open(outdoor_path) as img:
        res = validator.validate_image(img)
    print(f"\n[Test D - Ordinary Photograph: {outdoor_path.name}]")
    print(f"  Score = {res['domain_score']:.4f} (Threshold = {res['threshold']:.4f})")
    print(f"  Is Endoscopy = {res['is_endoscopy']} -> {'REJECTED (PASS)' if not res['is_endoscopy'] else 'ACCEPTED (FAIL)'}")
    assert not res["is_endoscopy"], f"Expected REJECT, got ACCEPT for outdoor photo (score {res['domain_score']})"

    # ── Test E: Known GastroVision Endoscopy Image -> ACCEPT ─────────────────
    known_endo_path = GASTROVISION_DIR / "Normal mucosa and vascular pattern in the large bowel"
    known_endo_file = sorted(list(known_endo_path.glob("*.jpg")))[0]
    with Image.open(known_endo_file) as img:
        res = validator.validate_image(img)
    print(f"\n[Test E - Known GastroVision Mucosa: {known_endo_file.name}]")
    print(f"  Score = {res['domain_score']:.4f} (Threshold = {res['threshold']:.4f})")
    print(f"  Is Endoscopy = {res['is_endoscopy']} -> {'ACCEPTED (PASS)' if res['is_endoscopy'] else 'REJECTED (FAIL)'}")
    assert res["is_endoscopy"], f"Expected ACCEPT, got REJECT for endoscopy image (score {res['domain_score']})"

    # ── Test F: Cancer Test Image -> ACCEPT -> Classifier runs ───────────────
    cancer_test_path = TEST_DATA_DIR / "colorectal_cancer" / "0c7db439-d40d-468f-9cc7-97a2a526dd7c.jpg"
    if not cancer_test_path.exists():
        cancer_test_path = PROJECT_ROOT / "app" / "samples" / "sample_colorectal_cancer.jpg"
    assert cancer_test_path.exists(), f"Cancer test file not found: {cancer_test_path}"
    with Image.open(cancer_test_path) as img:
        res = validator.validate_image(img)
        assert res["is_endoscopy"], f"Cancer test image failed validator: {res}"
        # Now run classifier
        t = eval_tf(img.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(classifier(t), dim=1)[0]
    print(f"\n[Test F - Cancer Test Image: {cancer_test_path.name}]")
    print(f"  Validator: Score = {res['domain_score']:.4f} -> ACCEPTED (PASS)")
    print(f"  Classifier: P(Cancer) = {probs[1]:.4f} ({probs[1]*100:.2f}%) | P(Diverticula) = {probs[0]:.4f}")
    assert probs[1] > 0.5, "Classifier failed to identify cancer"

    # ── Test G: Diverticula Test Image -> ACCEPT -> Classifier runs ──────────
    divert_test_path = TEST_DATA_DIR / "colon_diverticula" / "04675a9b-4858-439c-9199-94426a1e76a5.jpg"
    if not divert_test_path.exists():
        divert_test_path = PROJECT_ROOT / "app" / "samples" / "sample_colon_diverticula.jpg"
    assert divert_test_path.exists(), f"Diverticula test file not found: {divert_test_path}"
    with Image.open(divert_test_path) as img:
        res = validator.validate_image(img)
        assert res["is_endoscopy"], f"Diverticula test image failed validator: {res}"
        # Now run classifier
        t = eval_tf(img.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(classifier(t), dim=1)[0]
    print(f"\n[Test G - Diverticula Test Image: {divert_test_path.name}]")
    print(f"  Validator: Score = {res['domain_score']:.4f} -> ACCEPTED (PASS)")
    print(f"  Classifier: P(Diverticula) = {probs[0]:.4f} ({probs[0]*100:.2f}%) | P(Cancer) = {probs[1]:.4f}")
    assert probs[0] > 0.5, "Classifier failed to identify diverticula"

    print("\n" + "=" * 70)
    print("ALL TESTS (A, B, C, D, E, F, G) PASSED PERFECTLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
