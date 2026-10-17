"""
Integration Tests for SatQueryEngine and End-to-End Pipeline Orchestration.

Verifies:
- Land Cover classification workflow (Milestone 2 regression).
- Change Detection workflow with valid temporal pair inputs (Milestone 3).
- Configurable routing policies: 'provisional' vs 'clarify'.
- Incompatible input handling (single image for temporal task).
- Explicit unavailable stub for unsupported tasks (e.g. CHANGE_VQA).
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
    assert "SUCCESS" in response.status
    assert response.execution_policy in ("PROVISIONAL_EXECUTION", "STANDARD_EXECUTION")
    assert len(response.execution_trace) >= 5
    assert any("[Trace 1/5]" in t for t in response.execution_trace)
    assert any("[Trace 5/5]" in t for t in response.execution_trace)
    assert len(response.evidence.evidence_points) >= 2
    assert "Predicted land-cover" in response.answer or "No registered" in response.answer

    # Verify JSON serializability
    data = response.to_dict()
    assert json.dumps(data) is not None


def test_orchestrator_change_detection_flow(engine):
    query = "What changed in this area between 2024 and 2026?"
    t1 = np.zeros((30, 30, 3), dtype=np.float32)
    t2 = np.zeros((30, 30, 3), dtype=np.float32)
    # Inject change
    t2[10:20, 10:20, :] = 1.0

    response = engine.process_query(query, input_data=(t1, t2))

    assert isinstance(response, SatQueryResponse)
    assert response.routed_task == "CHANGE_DETECTION"
    assert "SUCCESS" in response.status
    assert len(response.evidence.evidence_points) >= 3
    assert any("Valid pixel coverage" in pt for pt in response.evidence.evidence_points)
    assert any("Largest connected change cluster" in pt for pt in response.evidence.evidence_points)


def test_orchestrator_clarification_policy():
    engine_clarify = SatQueryEngine.create_default(routing_policy="clarify")
    query = "What changed between 2024 and 2026?"

    # With routing confidence below 0.65 and policy='clarify', execution halts
    response = engine_clarify.process_query(query, input_data=None)

    assert isinstance(response, SatQueryResponse)
    assert response.status == "NEEDS_CLARIFICATION"
    assert response.execution_policy == "CLARIFICATION_REQUIRED"
    assert "CLARIFICATION REQUIRED" in response.answer
    assert "tentatively identified" in response.answer


def test_orchestrator_unsupported_task_stub(engine):
    # Query routed to unsupported VQA stub
    query = "How many distinct water bodies are visible in this scene?"
    response = engine.process_query(query, input_data=None)

    assert isinstance(response, SatQueryResponse)
    assert response.routed_task in ("CHANGE_VQA", "SINGLE_IMAGE_VQA")
    assert response.status == "UNAVAILABLE_SPECIALIST"
    assert "No active specialist model" in response.answer
    assert "Milestone 3" in response.answer
    assert len(response.evidence.detected_classes) == 0  # No fabricated predictions


def test_orchestrator_incompatible_input_change_detection(engine):
    # Change query but provided with single image array
    query = "Detect land-cover modifications between these two acquisition dates."
    single_img = np.zeros((30, 30, 3), dtype=np.float32)

    response = engine.process_query(query, input_data=single_img)

    assert isinstance(response, SatQueryResponse)
    assert response.routed_task == "CHANGE_DETECTION"
    assert response.status == "INCOMPATIBLE_INPUT"
    assert "Incompatible input" in response.answer


def test_orchestrator_incompatible_input_land_cover(engine):
    query = "Classify the land cover across this satellite patch."
    response = engine.process_query(query, input_data=np.zeros(20))

    assert isinstance(response, SatQueryResponse)
    assert response.routed_task == "LAND_COVER_CLASSIFICATION"
    assert response.status == "INCOMPATIBLE_INPUT"
    assert "Incompatible input" in response.answer
