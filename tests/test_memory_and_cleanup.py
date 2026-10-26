#!/usr/bin/env python3
"""
Automated Memory Diagnostics & Sequential Model Loading Test Suite
===================================================================
Validates strict 512 MB memory constraints for Render Free deployment:
1. Sequential comparative inference never retains both models simultaneously.
2. Models and intermediate Grad-CAM tensors are immediately released.
3. EndoscopyValidator extractor is not left resident in RAM.
4. Peak memory remains safely under 500 MB throughout comparison.
5. In-memory history does not store unbounded multi-megabyte image strings.
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.api import app, _MODELS_CACHE, _VALIDATOR, _ANALYSIS_HISTORY, get_current_rss_mb


@pytest.fixture
def client():
    return TestClient(app)


def test_compare_sequential_memory_cleanup(client):
    """
    Ensures POST /api/compare executes sequentially, releases model weights,
    and leaves the process with zero cached models and zero resident validator extractor.
    """
    initial_rss = get_current_rss_mb()
    print(f"\n[Memory Test] Initial RSS: {initial_rss} MB")

    r = client.post("/api/compare", data={"sample_id": "cancer"})
    assert r.status_code == 200, f"Compare failed: {r.text}"
    data = r.json()

    # 1. Verify response structure
    assert "consensus" in data
    assert "models" in data
    assert len(data["models"]) == 2
    assert "densenet121" in data["models"]
    assert "resnet50" in data["models"]

    # 2. Verify both models ran successfully with Grad-CAM
    for m_key in ["densenet121", "resnet50"]:
        res = data["models"][m_key]
        assert res["status"] == "success", f"Model {m_key} failed: {res.get('error')}"
        assert res["prediction"] in ("Colorectal Carcinoma", "Colon Diverticula")
        assert res["confidence"] > 0.5
        assert "overlay_uri" in res and res["overlay_uri"].startswith("data:image/jpeg;base64,")
        assert "heatmap_uri" in res and res["heatmap_uri"].startswith("data:image/jpeg;base64,")

    # 3. Verify memory diagnostics
    assert "memory_diagnostics" in data
    diag = data["memory_diagnostics"]
    print(f"[Memory Test] Compare Peak RSS: {diag['peak_rss_mb']} MB (Limit: {diag['limit_mb']} MB)")
    print(f"[Memory Test] Compare Final RSS: {diag['final_rss_mb']} MB")

    # Strict Render 512MB limit check
    assert diag["peak_rss_mb"] < 495.0, f"Peak RSS exceeded safe 512MB headroom: {diag['peak_rss_mb']} MB"

    # 4. Verify post-request state: zero cached models
    assert len(_MODELS_CACHE) == 0, f"_MODELS_CACHE was not cleared: {list(_MODELS_CACHE.keys())}"

    # 5. Verify validator extractor is NOT resident
    if _VALIDATOR is not None:
        assert getattr(_VALIDATOR, "_extractor", None) is None, "Validator extractor still resident in memory!"


def test_analyze_memory_cleanup(client):
    """
    Ensures POST /api/analyze cleans up its loaded model and does not retain it in _MODELS_CACHE.
    """
    for model_choice in ["densenet121", "resnet50"]:
        r = client.post("/api/analyze", data={
            "model_choice": model_choice,
            "sample_id": "cancer",
            "alpha": "0.45",
            "colormap": "jet",
        })
        assert r.status_code == 200, f"Analyze {model_choice} failed: {r.text}"
        res = r.json()
        assert res["prediction"] == "Colorectal Cancer"
        assert len(res["overlay_uri"]) > 100

        # Model must be evicted and freed immediately after request
        assert len(_MODELS_CACHE) == 0, f"Model {model_choice} retained in _MODELS_CACHE after /api/analyze"


def test_no_unbounded_history_images(client):
    """
    Ensures session history entries do not store multi-megabyte base64 strings
    that would permanently inflate memory.
    """
    # Trigger 2 analyses
    client.post("/api/analyze", data={"model_choice": "densenet121", "sample_id": "cancer"})
    client.post("/api/analyze", data={"model_choice": "resnet50", "sample_id": "diverticula"})

    r = client.get("/api/history")
    assert r.status_code == 200
    hist = r.json().get("history", [])
    assert len(hist) <= 5

    for entry in hist:
        # Crucial: base64 image data must not be stored in global history list
        assert "original_uri" not in entry, "original_uri found in global history entry!"
        assert "heatmap_uri" not in entry, "heatmap_uri found in global history entry!"
        assert "overlay_uri" not in entry, "overlay_uri found in global history entry!"
        assert "prediction" in entry
        assert "model_name" in entry


def test_health_reports_memory(client):
    """
    Ensures GET /api/health includes current RSS memory diagnostics.
    """
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert "memory" in data
    assert "rss_mb" in data["memory"]
    assert data["memory"]["rss_mb"] > 0
    assert data["memory"]["limit_mb"] == 512
    print(f"\n[Health Check] Reported RSS: {data['memory']['rss_mb']} MB / 512 MB")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
