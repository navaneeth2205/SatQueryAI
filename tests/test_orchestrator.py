"""
Integration Tests for SatQueryEngine and End-to-End Pipeline Orchestration.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from satquery.orchestrator import SatQueryEngine, SatQueryResponse


@pytest.fixture
def engine():
    return SatQueryEngine.create_default()


@pytest.fixture
def sample_feature_vector():
    # 14 S1 + 70 S2 realistic synthetic values
    s1 = np.full(14, -18.0, dtype=np.float32)
    s2 = np.full(70, 450.0, dtype=np.float32)
    return np.concatenate([s1, s2])


def test_orchestrator_land_cover_flow(engine, sample_feature_vector):
    query = "What land-cover types are visible in this image?"
    response = engine.process_query(query, input_data=sample_feature_vector)

    assert isinstance(response, SatQueryResponse)
    assert response.routed_task == "LAND_COVER_CLASSIFICATION"
    assert response.status == "SUCCESS"
    assert len(response.execution_trace) >= 5
    assert any("[Trace 1/5]" in t for t in response.execution_trace)
    assert any("[Trace 5/5]" in t for t in response.execution_trace)
    assert len(response.evidence.evidence_points) >= 2
    assert "Detected land-cover" in response.answer or "No registered" in response.answer

    # Verify JSON serializability
    data = response.to_dict()
    assert json.dumps(data) is not None


def test_orchestrator_unsupported_task_stub(engine):
    # Query clearly requesting change detection
    query = "What changed between 2024 and 2026?"
    response = engine.process_query(query, input_data=None)

    assert isinstance(response, SatQueryResponse)
    assert response.routed_task == "CHANGE_DETECTION"
    assert response.status == "UNAVAILABLE_SPECIALIST"
    assert "No active specialist model" in response.answer
    assert "Milestone 3" in response.answer
    assert len(response.evidence.detected_classes) == 0  # No fabricated predictions
    assert len(response.execution_trace) >= 5


def test_orchestrator_incompatible_input(engine):
    # Valid land cover query, but wrong feature length
    query = "Classify the land cover across this satellite patch."
    response = engine.process_query(query, input_data=np.zeros(20))

    assert isinstance(response, SatQueryResponse)
    assert response.routed_task == "LAND_COVER_CLASSIFICATION"
    assert response.status == "INCOMPATIBLE_INPUT"
    assert "Incompatible input" in response.answer
