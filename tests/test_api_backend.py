#!/usr/bin/env python3
"""
Test harness for FastAPI backend endpoints.
Verifies health, samples, validation, inference, Grad-CAM, model comparison, and PDF generation.
"""

import sys
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.api import app, SAMPLES_DIR

client = TestClient(app)

def test_backend_api():
    print("=" * 60)
    print("Testing ColonVision AI FastAPI Backend Endpoints")
    print("=" * 60)

    # 1. Health check
    print("\n1. Testing GET /api/health...")
    r = client.get("/api/health")
    assert r.status_code == 200, f"Health check failed: {r.text}"
    health = r.json()
    assert health["status"] == "online"
    assert "densenet121" in health["models"]
    assert "resnet50" in health["models"]
    assert "efficientnet_b0" in health["models"]
    print(f"   [PASS] Health check OK. Models available: {list(health['models'].keys())}")

    # 2. Research samples
    print("\n2. Testing GET /api/samples...")
    r = client.get("/api/samples")
    assert r.status_code == 200, f"Samples check failed: {r.text}"
    samples = r.json().get("samples", [])
    assert len(samples) >= 2, f"Expected at least 2 samples, got {len(samples)}"
    for s in samples:
        assert "image_url" in s, f"Missing image_url in sample {s['id']}"
        assert "static_url" in s, f"Missing static_url in sample {s['id']}"
        assert s.get("available") is True, f"Sample {s['id']} should be available"
    print(f"   [PASS] Loaded {len(samples)} research samples: {[s['id'] for s in samples]}")

    # 2b. Direct sample image endpoints
    print("\n2b. Testing GET /api/samples/image/{sample_id}...")
    for s_id in ["cancer", "diverticula"]:
        r_img = client.get(f"/api/samples/image/{s_id}")
        assert r_img.status_code == 200, f"Sample image {s_id} failed: {r_img.status_code}"
        assert "image/jpeg" in r_img.headers["content-type"]
        assert len(r_img.content) > 10000, f"Sample image {s_id} too small: {len(r_img.content)} bytes"
        print(f"   [PASS] Sample image endpoint for '{s_id}' served {len(r_img.content)} bytes.")

    # 2c. Direct static file route
    print("\n2c. Testing GET /samples/... static files...")
    for fn in ["sample_colorectal_cancer.jpg", "sample_colon_diverticula.jpg"]:
        r_stat = client.get(f"/samples/{fn}")
        assert r_stat.status_code == 200, f"Static sample {fn} failed: {r_stat.status_code}"
        assert len(r_stat.content) > 10000
        print(f"   [PASS] Static sample '/samples/{fn}' served {len(r_stat.content)} bytes.")

    # 2d. Unknown sample 404 JSON error
    r_bad_sample = client.get("/api/samples/image/unknown_sample_id")
    assert r_bad_sample.status_code == 404
    assert "application/json" in r_bad_sample.headers["content-type"]
    assert "error" in r_bad_sample.json()
    print("   [PASS] Nonexistent sample returned structured 404 JSON.")

    # 3. Domain validation
    print("\n3. Testing POST /api/validate with Cancer sample...")
    r = client.post("/api/validate", data={"sample_id": "cancer"})
    assert r.status_code == 200, f"Validation failed: {r.text}"
    val_data = r.json()
    assert val_data["is_endoscopy"] is True
    print(f"   [PASS] Endoscopy validated. Score: {val_data['domain_score']:.4f} >= {val_data['threshold']:.4f}")

    # 4. Clinical inference with DenseNet-121
    print("\n4. Testing POST /api/analyze with DenseNet-121...")
    r = client.post("/api/analyze", data={
        "model_choice": "densenet121",
        "sample_id": "cancer",
        "alpha": "0.45",
        "colormap": "jet",
    })
    assert r.status_code == 200, f"Analysis failed: {r.text}"
    analysis = r.json()
    assert analysis["prediction"] == "Colorectal Cancer"
    assert analysis["confidence"] > 0.8
    assert "overlay_uri" in analysis
    assert "heatmap_uri" in analysis
    print(f"   [PASS] Inference OK: {analysis['prediction']} ({analysis['confidence_pct']})")
    print(f"          Peak Saliency: (x:{analysis['peak_x']}, y:{analysis['peak_y']})")
    print(f"          Target Layer: {analysis['target_layer']}, Latency: {analysis['latency_ms']} ms")

    # 5. Multi-model comparative inference
    print("\n5. Testing POST /api/compare...")
    r = client.post("/api/compare", data={"sample_id": "cancer"})
    assert r.status_code == 200, f"Comparison failed: {r.text}"
    compare_data = r.json()
    assert "consensus" in compare_data
    assert "models" in compare_data
    assert len(compare_data["models"]) == 3
    for m_key in ["densenet121", "resnet50", "efficientnet_b0"]:
        assert m_key in compare_data["models"], f"Missing {m_key} in comparison results"
        m_res = compare_data["models"][m_key]
        assert m_res["status"] == "success", f"Model {m_key} did not succeed: {m_res}"
        assert "overlay_uri" in m_res and m_res["overlay_uri"].startswith("data:image/jpeg;base64,"), f"Invalid overlay for {m_key}"
        assert "confidence" in m_res and m_res["confidence"] > 0
        assert "latency_ms" in m_res and m_res["latency_ms"] > 0
        assert "attribution_layer" in m_res and len(m_res["attribution_layer"]) > 0
    print(f"   [PASS] Consensus: {compare_data['consensus']['summary']}")
    print(f"          Concordance kappa: {compare_data['consensus']['kappa']}")

    # 6. PDF Report Generation
    print("\n6. Testing POST /api/report...")
    report_payload = {
        "model_name": analysis["model_name"],
        "prediction": analysis["prediction"],
        "confidence": analysis["confidence"],
        "prob_cancer": analysis["probabilities"]["colorectal_cancer"],
        "prob_diverticula": analysis["probabilities"]["colon_diverticula"],
        "validation_score": analysis["domain_score"],
        "threshold": analysis["threshold"],
        "original_uri": analysis["original_uri"],
        "heatmap_uri": analysis["heatmap_uri"],
        "overlay_uri": analysis["overlay_uri"],
    }
    r = client.post("/api/report", json=report_payload)
    assert r.status_code == 200, f"Report failed: {r.text}"
    assert r.headers["content-type"] == "application/pdf"
    assert len(r.content) > 3000, f"PDF too small: {len(r.content)} bytes"
    print(f"   [PASS] PDF report generated successfully ({len(r.content)} bytes)")

    # 7. Retrospective History and Clear History
    print("\n7. Testing GET /api/history and DELETE /api/history...")
    r = client.get("/api/history")
    assert r.status_code == 200
    hist = r.json().get("history", [])
    assert len(hist) >= 1, "Expected at least 1 recorded analysis"
    print(f"   [PASS] History retrieved ({len(hist)} item(s))")

    r_del = client.delete("/api/history")
    assert r_del.status_code == 200
    assert r_del.json().get("status") == "cleared"
    r_empty = client.get("/api/history")
    assert len(r_empty.json().get("history", [])) == 0
    print("   [PASS] History cleared successfully (0 items remaining)")

    # 8. Testing ResNet-50 and EfficientNet-B0 inference
    print("\n8. Testing inference on ResNet-50 and EfficientNet-B0...")
    for m_key in ["resnet50", "efficientnet_b0"]:
        r_m = client.post("/api/analyze", data={
            "model_choice": m_key,
            "sample_id": "diverticula",
            "alpha": "0.45",
            "colormap": "magma",
        })
        assert r_m.status_code == 200, f"Analysis for {m_key} failed: {r_m.text}"
        assert "application/json" in r_m.headers["content-type"]
        assert "overlay_uri" in r_m.json()
        print(f"   [PASS] {m_key}: {r_m.json()['prediction']} ({r_m.json()['confidence_pct']})")

    # 9. Verify Error Handling & JSON Guarantee (Issue 1)
    print("\n9. Testing API Error Handling and Content-Type Guarantees...")
    # 9a. Unknown API route must return 404 JSON, NOT HTML
    r_404 = client.get("/api/nonexistent_route")
    assert r_404.status_code == 404
    assert "application/json" in r_404.headers["content-type"]
    assert not r_404.text.startswith("<!DOCTYPE")
    print("   [PASS] Unknown /api endpoint returned JSON 404 (no HTML).")

    # 9b. Invalid model parameter returns 400 JSON
    r_bad_model = client.post("/api/analyze", data={"model_choice": "nonexistent_model", "sample_id": "cancer"})
    assert r_bad_model.status_code == 400
    assert "application/json" in r_bad_model.headers["content-type"]
    print("   [PASS] Invalid model returned JSON 400 error.")

    # 9c. Missing file/sample returns 400 JSON
    r_missing = client.post("/api/analyze", data={"model_choice": "densenet121"})
    assert r_missing.status_code == 400
    assert "application/json" in r_missing.headers["content-type"]
    print("   [PASS] Missing image input returned JSON 400 error.")

    # 9d. Root route serves compiled SPA HTML
    r_root = client.get("/")
    assert r_root.status_code == 200
    assert "text/html" in r_root.headers["content-type"]
    assert "ColonVision AI" in r_root.text
    print("   [PASS] Root route successfully served compiled frontend SPA.")

    print("\n" + "=" * 60)
    print("ALL FASTAPI BACKEND ENDPOINTS AND ERROR TESTS PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    test_backend_api()
