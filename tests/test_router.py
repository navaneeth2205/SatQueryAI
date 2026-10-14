"""
Tests for TaskRouter Engine.
"""

import tempfile
from pathlib import Path
import pytest
from satquery.router import TaskRouter, ExecutionRoute


@pytest.fixture
def sample_data():
    queries = [
        "What type of land cover is visible?",
        "Classify the vegetation and forest terrain.",
        "Identify arable land or woodland.",
        "What changed between 2024 and 2026?",
        "Detect new urban construction between image 1 and image 2.",
        "Highlight land alterations over time.",
        "Combine optical imagery and radar backscatter observations.",
        "Compare SAR and optical bands for soil moisture.",
        "Cross-examine optical multispectral and SAR observations."
    ]
    labels = [
        "LAND_COVER_CLASSIFICATION",
        "LAND_COVER_CLASSIFICATION",
        "LAND_COVER_CLASSIFICATION",
        "CHANGE_DETECTION",
        "CHANGE_DETECTION",
        "CHANGE_DETECTION",
        "OPTICAL_SAR_FUSION",
        "OPTICAL_SAR_FUSION",
        "OPTICAL_SAR_FUSION"
    ]
    return queries, labels


def test_router_fit_and_predict(sample_data):
    queries, labels = sample_data
    router = TaskRouter(confidence_threshold=0.60)
    router.fit(queries, labels)

    assert router.is_fitted
    assert len(router.classes_) == 3

    route = router.predict("What is the land-cover class?")
    assert isinstance(route, ExecutionRoute)
    assert route.task == "LAND_COVER_CLASSIFICATION"
    assert route.required_input == "single_image"
    assert 0.0 <= route.confidence <= 1.0
    assert len(route.execution_trace) == 5
    assert len(route.class_probabilities) == 3

    # Check probabilities sum to approx 1.0
    total_prob = sum(route.class_probabilities.values())
    assert abs(total_prob - 1.0) < 0.05


def test_confidence_threshold_flag(sample_data):
    queries, labels = sample_data
    # Set impossible threshold to force review flag
    router = TaskRouter(confidence_threshold=0.999)
    router.fit(queries, labels)

    route = router.predict("Can you find something?")
    assert route.requires_review is True


def test_router_save_and_load(sample_data):
    queries, labels = sample_data
    router = TaskRouter(confidence_threshold=0.70)
    router.fit(queries, labels)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.joblib"
        router.save(model_path)
        assert model_path.exists()

        loaded_router = TaskRouter.load(model_path)
        assert loaded_router.is_fitted
        assert loaded_router.confidence_threshold == 0.70

        test_query = "What changed between 2024 and 2026?"
        original_route = router.predict(test_query)
        loaded_route = loaded_router.predict(test_query)

        assert loaded_route.task == original_route.task
        assert loaded_route.confidence == original_route.confidence
        assert loaded_route.task == "CHANGE_DETECTION"
