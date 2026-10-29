"""
Integration tests for SatQuery AI Web Demo Application.

Tests Flask routes:
- GET / (HTML Analyst Dashboard)
- GET /api/catalog (Catalog metadata, preset queries, sample datasets)
- POST /api/query (Interactive end-to-end routing and specialist execution)
"""

import json
import pytest
from satquery.demo.app import create_app


@pytest.fixture
def client():
    """Create Flask test client."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_route(client):
    """Test that the main dashboard HTML is served properly."""
    res = client.get("/")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert "SatQuery AI" in html
    assert "Earth Observation Intelligence" in html


def test_catalog_endpoint(client):
    """Test /api/catalog returns expected datasets, presets, and counts."""
    res = client.get("/api/catalog")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "preset_queries" in data
    assert len(data["preset_queries"]) > 0
    assert "change_detection_scenes" in data
    assert "land_cover_samples" in data
    assert data["total_cd_scenes"] >= 10


def test_query_endpoint_empty_query(client):
    """Test /api/query rejects empty queries."""
    res = client.post("/api/query", json={"query": ""})
    assert res.status_code == 400
    data = res.get_json()
    assert "error" in data


def test_query_endpoint_unsupported_task(client):
    """Test /api/query handles unsupported tasks with honest UNAVAILABLE status."""
    payload = {
        "query": "Did the reservoir surface shrink between the two dates?",
        "mode": "query_only"
    }
    res = client.post("/api/query", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert "response" in data
    assert data["response"]["status"] == "UNAVAILABLE_SPECIALIST"
    assert data["response"]["routed_task"] in ("CHANGE_VQA", "SINGLE_IMAGE_VQA", "SCENE_CAPTIONING", "VISUAL_GROUNDING")


def test_query_endpoint_change_detection_real_scene(client):
    """Test /api/query with a real OSCD scene pair and verify visuals."""
    payload = {
        "query": "What changed in this area between the acquisition dates?",
        "mode": "change_detection",
        "cd_sample_idx": 0,
        "synthetic_identical": False
    }
    res = client.post("/api/query", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert "response" in data
    assert data["response"]["status"] in ("SUCCESS", "PROVISIONAL_SUCCESS")
    assert data["response"]["routed_task"] == "CHANGE_DETECTION"
    
    # Verify visual assets are generated
    assert "visuals" in data
    assert "t1_image" in data["visuals"]
    assert "t2_image" in data["visuals"]
    assert "change_overlay" in data["visuals"]
    assert data["visuals"]["change_overlay"].startswith("data:image/png;base64,")


def test_query_endpoint_change_detection_synthetic_identical(client):
    """Test /api/query with synthetic identical image pair returns zero change."""
    payload = {
        "query": "Detect surface modifications across these two dates.",
        "mode": "change_detection",
        "cd_sample_idx": 0,
        "synthetic_identical": True
    }
    res = client.post("/api/query", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert "response" in data
    assert data["response"]["status"] in ("SUCCESS", "PROVISIONAL_SUCCESS")
    evidence = data["response"]["evidence"]
    assert evidence["sensor_feature_summary"]["changed_area_pct_of_valid"] == 0.0


def test_query_endpoint_land_cover_sample(client):
    """Test /api/query with land-cover classification sample."""
    payload = {
        "query": "What land-cover classes are present in this satellite scene?",
        "mode": "land_cover",
        "sample_idx": 0
    }
    res = client.post("/api/query", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert "response" in data
    assert data["response"]["status"] in ("SUCCESS", "PROVISIONAL_SUCCESS")
    assert data["response"]["routed_task"] == "LAND_COVER_CLASSIFICATION"
    assert "detected_classes" in data["response"]["evidence"]
    assert "class_probabilities" in data["response"]["evidence"]



def test_guard_banner_presence_in_html(client):
    """Test that the Zero-Hallucination Guard and algorithm selectors are present in HTML."""
    res = client.get("/")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert "guardBanner" in html
    assert "Zero-Hallucination Specialist Guard Active" in html
    assert "cdMethodSelect" in html
    assert "lcModelSelect" in html


def test_query_endpoint_change_detection_algorithm_options(client):
    """Test /api/query respects cd_method options (smoothed, hysteresis, standard)."""
    for method in ("smoothed", "hysteresis", "standard"):
        payload = {
            "query": "What changed in this area between the acquisition dates?",
            "mode": "change_detection",
            "cd_sample_idx": 0,
            "cd_method": method
        }
        res = client.post("/api/query", json=payload)
        assert res.status_code == 200
        data = res.get_json()
        assert data["response"]["status"] in ("SUCCESS", "PROVISIONAL_SUCCESS")
        assert data["visuals"]["cd_method"] == method
        assert data["visuals"]["change_overlay"].startswith("data:image/png;base64,")


def test_query_endpoint_land_cover_model_options(client):
    """Test /api/query respects model_version options (phase2 vs baseline)."""
    for model_ver in ("phase2", "baseline"):
        payload = {
            "query": "Classify land cover types in this scene.",
            "mode": "land_cover",
            "sample_idx": 0,
            "model_version": model_ver
        }
        res = client.post("/api/query", json=payload)
        assert res.status_code == 200
        data = res.get_json()
        assert data["response"]["status"] in ("SUCCESS", "PROVISIONAL_SUCCESS")
        assert "active_model" in data["visuals"]
