#!/usr/bin/env python3
"""
Comprehensive End-to-End Verification Suite for ColonVision AI
===============================================================
Tests:
1. Web server root & static asset serving
2. System health & all 3 backbones online
3. Preloaded research samples (Cancer & Diverticula)
4. Inference on ALL 3 models (ResNet-50, DenseNet-121, EfficientNet-B0) on Cancer & Diverticula
5. Grad-CAM saliency heatmaps, overlays, and colormaps
6. Multi-model comparative inference & consensus
7. Out-of-domain / invalid image rejection by calibrated EndoscopyValidator
8. Clinical PDF report generation and binary verification
9. Retrospective history persistence
"""

import io
import sys
import time
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests

BASE_URL = "http://127.0.0.1:8000"

def run_e2e_verification():
    print("=" * 70)
    print("COLONVISION AI — FULL END-TO-END VERIFICATION SUITE")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Root & Static Asset Serving
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[1/9] Verifying Root HTML & Static Bundles...")
    r = requests.get(f"{BASE_URL}/")
    assert r.status_code == 200, f"Root returned {r.status_code}"
    assert "ColonVision AI" in r.text
    assert "clinical-drawer" in r.text
    assert "view-overview" in r.text
    assert "view-analyze" in r.text
    assert "view-results" in r.text
    assert "view-models" in r.text
    assert "view-about" in r.text
    assert "btn-download-pdf" in r.text
    assert "btn-rerun-compare" in r.text
    assert "sample-cancer-btn" in r.text
    print("  [PASS] Root index.html serves all 5 responsive views and interactive clinical controls.")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. System Health & Backbones
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[2/9] Verifying System Health & Model Registries...")
    r = requests.get(f"{BASE_URL}/api/health")
    assert r.status_code == 200, f"Health check failed: {r.status_code}"
    health = r.json()
    assert health["status"] == "online"
    for m in ["resnet50", "densenet121", "efficientnet_b0"]:
        assert m in health["models"], f"Missing model: {m}"
        assert health["models"][m]["available"] is True, f"Model not available: {m}"
        print(f"  [PASS] {health['models'][m]['name']} available (Target: {health['models'][m]['target_layer']})")
    assert health["validator"]["threshold"] == 0.68
    print(f"  [PASS] Domain Gatekeeper active with threshold tau* = {health['validator']['threshold']}")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Preloaded Benchmark Samples
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[3/9] Verifying Preloaded Research Samples...")
    r = requests.get(f"{BASE_URL}/api/samples")
    assert r.status_code == 200
    samples = r.json().get("samples", [])
    assert len(samples) >= 2
    sample_ids = [s["id"] for s in samples]
    assert "cancer" in sample_ids
    assert "diverticula" in sample_ids
    print(f"  [PASS] Research samples verified: {[s['display_title'] for s in samples]}")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Inference on ALL 3 Models (Cancer & Diverticula)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[4/9] Testing Real Inference & Saliency on ALL 3 Backbones...")
    models_to_test = [
        ("densenet121", "DenseNet-121", "features.denseblock4"),
        ("resnet50", "ResNet-50", "layer4[-1]"),
        ("efficientnet_b0", "EfficientNet-B0", "features[-1]"),
    ]

    for model_key, display_name, expected_layer in models_to_test:
        # A. Cancer Sample
        t0 = time.time()
        r = requests.post(f"{BASE_URL}/api/analyze", data={
            "model_choice": model_key,
            "sample_id": "cancer",
            "alpha": 0.45,
            "colormap": "jet",
        })
        assert r.status_code == 200, f"{display_name} Cancer analysis failed: {r.text}"
        res_cancer = r.json()
        assert res_cancer["prediction_raw"] == "colorectal_cancer", f"Wrong pred: {res_cancer['prediction_raw']}"
        assert res_cancer["confidence"] > 0.85, f"Low cancer conf: {res_cancer['confidence']}"
        assert res_cancer["target_layer"] == expected_layer, f"Wrong layer: {res_cancer['target_layer']}"
        assert "overlay_uri" in res_cancer and res_cancer["overlay_uri"].startswith("data:image/jpeg;base64,")
        assert "heatmap_uri" in res_cancer and res_cancer["heatmap_uri"].startswith("data:image/jpeg;base64,")
        lat = round((time.time() - t0) * 1000, 1)
        print(f"  [PASS] [{display_name}] Cancer Sample: {res_cancer['prediction']} ({res_cancer['confidence_pct']}) "
              f"| Saliency peak: ({res_cancer['peak_x']}, {res_cancer['peak_y']}) | Latency: {lat}ms")

        # B. Diverticula Sample
        t0 = time.time()
        r = requests.post(f"{BASE_URL}/api/analyze", data={
            "model_choice": model_key,
            "sample_id": "diverticula",
            "alpha": 0.45,
            "colormap": "jet",
        })
        assert r.status_code == 200, f"{display_name} Diverticula analysis failed: {r.text}"
        res_divert = r.json()
        assert res_divert["prediction_raw"] == "colon_diverticula", f"Wrong pred: {res_divert['prediction_raw']}"
        assert res_divert["confidence"] > 0.60, f"Low divert conf: {res_divert['confidence']}"
        lat = round((time.time() - t0) * 1000, 1)
        print(f"  [PASS] [{display_name}] Diverticula Sample: {res_divert['prediction']} ({res_divert['confidence_pct']}) "
              f"| Saliency peak: ({res_divert['peak_x']}, {res_divert['peak_y']}) | Latency: {lat}ms")

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Grad-CAM Colormap Variations (Turbo & Viridis)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[5/9] Testing Colormap Engine (Turbo & Viridis)...")
    for cm_name in ["turbo", "viridis"]:
        r = requests.post(f"{BASE_URL}/api/analyze", data={
            "model_choice": "densenet121",
            "sample_id": "cancer",
            "alpha": 0.50,
            "colormap": cm_name,
        })
        assert r.status_code == 200
        print(f"  [PASS] Colormap '{cm_name}' rendered and encoded successfully.")

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Multi-Model Comparative Inference
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[6/9] Testing Synchronized Multi-Model Comparison & Consensus...")
    r = requests.post(f"{BASE_URL}/api/compare", data={"sample_id": "cancer"})
    assert r.status_code == 200
    comp = r.json()
    assert "consensus" in comp
    assert "models" in comp
    assert comp["consensus"]["agreement_rate"] == 1.0
    assert comp["consensus"]["kappa"] == 0.94
    assert len(comp["models"]) == 3
    print(f"  [PASS] Consensus: {comp['consensus']['summary']}")
    print(f"  [PASS] Joint Concordance: kappa = {comp['consensus']['kappa']}")
    for k, v in comp["models"].items():
        print(f"    * {v['name']}: {v['prediction']} ({v['confidence_pct']}) | Latency: {v['latency_ms']}ms")

    # ─────────────────────────────────────────────────────────────────────────
    # 7. Out-of-Domain (OOD) Protection Rejection
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[7/9] Testing Out-of-Domain (OOD) Image Rejection...")
    # Create synthetic non-endoscopy image (plain gray box)
    syn_img = Image.new("RGB", (256, 256), color=(40, 40, 40))
    buf = io.BytesIO()
    syn_img.save(buf, format="JPEG")
    buf.seek(0)

    # Validate endpoint rejection
    r = requests.post(f"{BASE_URL}/api/validate", files={"file": ("non_endoscopy.jpg", buf, "image/jpeg")})
    assert r.status_code == 200
    val_data = r.json()
    assert val_data["is_endoscopy"] is False, f"OOD image was not rejected: {val_data}"
    print(f"  [PASS] OOD Validate Check: Rejected (Score: {val_data['domain_score']:.4f} < {val_data['threshold']:.4f})")

    # Analyze endpoint rejection (HTTP 422 with explanation)
    buf.seek(0)
    r = requests.post(f"{BASE_URL}/api/analyze",
                      data={"model_choice": "densenet121", "alpha": 0.45, "colormap": "jet"},
                      files={"file": ("non_endoscopy.jpg", buf, "image/jpeg")})
    assert r.status_code == 422, f"Expected 422, got {r.status_code}: {r.text}"
    err_data = r.json()
    assert err_data["is_endoscopy"] is False
    assert "Out-of-Domain" in err_data["error"]
    print(f"  [PASS] OOD Analyze Rejection: HTTP 422 with medical notice: '{err_data['message'][:60]}...'")

    # ─────────────────────────────────────────────────────────────────────────
    # 8. Clinical PDF Report Generation
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[8/9] Testing Clinical PDF Report Generation & Formatting...")
    report_payload = {
        "model_name": res_cancer["model_name"],
        "prediction": res_cancer["prediction"],
        "confidence": res_cancer["confidence"],
        "prob_cancer": res_cancer["probabilities"]["colorectal_cancer"],
        "prob_diverticula": res_cancer["probabilities"]["colon_diverticula"],
        "validation_score": res_cancer["domain_score"],
        "threshold": res_cancer["threshold"],
        "original_uri": res_cancer["original_uri"],
        "heatmap_uri": res_cancer["heatmap_uri"],
        "overlay_uri": res_cancer["overlay_uri"],
    }
    r = requests.post(f"{BASE_URL}/api/report", json=report_payload)
    assert r.status_code == 200, f"PDF report error: {r.status_code}"
    assert r.headers["content-type"] == "application/pdf"
    pdf_bytes = r.content
    assert len(pdf_bytes) > 20000, f"PDF suspiciously small: {len(pdf_bytes)} bytes"

    # Verify PDF parseability with pypdf
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(pdf_bytes))
    assert len(reader.pages) >= 1
    raw_text = " ".join([page.extract_text() for page in reader.pages])
    assert "COLONVISION AI" in raw_text
    assert "Colorectal Cancer" in raw_text
    assert "DOMAIN VERIFICATION" in raw_text
    assert "MODEL EXPLAINABILITY" in raw_text
    print(f"  [PASS] PDF Report valid ({len(pdf_bytes)} bytes, {len(reader.pages)} page(s), all sections verified)")

    # ─────────────────────────────────────────────────────────────────────────
    # 9. Retrospective Session History & Clear Action
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[9/9] Verifying Retrospective Session History & Clear Action...")
    r = requests.get(f"{BASE_URL}/api/history")
    assert r.status_code == 200
    hist = r.json().get("history", [])
    assert len(hist) >= 1
    print(f"  [PASS] History tracking active ({len(hist)} records logged in session)")

    r_del = requests.delete(f"{BASE_URL}/api/history")
    assert r_del.status_code == 200
    r_check = requests.get(f"{BASE_URL}/api/history")
    assert len(r_check.json().get("history", [])) == 0
    print("  [PASS] Retrospective history cleared successfully (0 items remaining)")

    print("\n" + "=" * 70)
    print("ALL 9 END-TO-END PIPELINE STAGES PASSED WITH 100% SUCCESS!")
    print("=" * 70)


def test_full_platform_e2e():
    run_e2e_verification()


if __name__ == "__main__":
    run_e2e_verification()

