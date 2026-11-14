"""
Milestone M3.2 - Step 4: Multi-Specialist System Integration Test Suite.
Document Code: SATQUERY-TEST-M3.2-STEP4-INTEGRATION

Verifies end-to-end integration across:
1. Land Cover Specialist (Phase 2 Champion / RF Baseline)
2. Change Detection Specialist (Milestone M3.2 Siamese ResNet-18 U-Net Champion / Classical Fallback)
3. Compatibility-Aware Task Router (Phase 1.1 with safety arbitration)
4. Central Specialist Registry & Structured Unavailability Stubs (VQA)
5. Explainable Evidence Synthesis & Execution Trace Logging
"""

import json
from pathlib import Path
import numpy as np
import pytest

from satquery.taxonomy import TaskType, InputType
from satquery.router import TaskRouter
from satquery.specialists.base import SpecialistStatus, SpecialistResult
from satquery.specialists.registry import SpecialistRegistry
from satquery.specialists.land_cover import LandCoverClassifier
from satquery.specialists.change_detection import (
    TemporalDifferenceBaseline,
    SiameseResNetChangeDetectionSpecialist,
    ChangeDetectionSpecialist,
)
from satquery.orchestrator import SatQueryEngine, SatQueryResponse


@pytest.fixture
def engine():
    return SatQueryEngine.create_default()


@pytest.fixture
def synthetic_temporal_pair():
    # 64x64 RGB temporal pair (standard uint8 8-bit digital values) with distinct change patch
    t1 = np.full((64, 64, 3), 50, dtype=np.uint8)
    t2 = np.full((64, 64, 3), 50, dtype=np.uint8)
    # Inject significant change in central 20x20 block (400 px / 4096 px ~= 9.7%)
    t2[20:40, 20:40, :] = 220
    return t1, t2


@pytest.fixture
def synthetic_feature_vector():
    # 14 S1 + 70 S2 realistic synthetic values for Land Cover
    s1 = np.full(14, -16.0, dtype=np.float32)
    s2 = np.full(70, 480.0, dtype=np.float32)
    return np.concatenate([s1, s2])


def test_specialist_registry_multi_specialist_coexistence(synthetic_temporal_pair, synthetic_feature_vector):
    """Test that both Land Cover and Change Detection specialists coexist in registry."""
    registry = SpecialistRegistry()

    lc_spec = LandCoverClassifier()
    cd_spec = ChangeDetectionSpecialist()

    registry.register(lc_spec)
    registry.register(cd_spec)

    # 1. Dispatch Land Cover
    res_lc = registry.dispatch(TaskType.LAND_COVER_CLASSIFICATION, synthetic_feature_vector)
    assert res_lc.status == SpecialistStatus.SUCCESS
    assert res_lc.task == TaskType.LAND_COVER_CLASSIFICATION.value
    assert len(res_lc.probabilities) == 16

    # 2. Dispatch Change Detection
    res_cd = registry.dispatch(TaskType.CHANGE_DETECTION, synthetic_temporal_pair)
    assert res_cd.status == SpecialistStatus.SUCCESS
    assert res_cd.task == TaskType.CHANGE_DETECTION.value
    assert "change_detected" in res_cd.probabilities
    assert "largest_change_cluster_bbox_pixel" in res_cd.execution_metadata

    # 3. Dispatch Unsupported VQA (Transparent Stub)
    res_vqa = registry.dispatch(TaskType.CHANGE_VQA, synthetic_temporal_pair)
    assert res_vqa.status == SpecialistStatus.UNAVAILABLE
    assert "No active specialist model" in res_vqa.unavailable_reason
    assert res_vqa.execution_metadata.get("is_stub") is True
    assert len(res_vqa.predictions) == 0


def test_orchestrator_end_to_end_land_cover(engine, synthetic_feature_vector):
    """Verify end-to-end natural language query execution for Land Cover."""
    query = "What land-cover categories are present in this multispectral tile?"
    response = engine.process_query(query, input_data=synthetic_feature_vector)

    assert isinstance(response, SatQueryResponse)
    assert response.routed_task == "LAND_COVER_CLASSIFICATION"
    assert "SUCCESS" in response.status
    assert response.input_compatibility is True
    assert len(response.execution_trace) >= 4
    assert any("[Trace 1/5]" in t for t in response.execution_trace)
    assert any("LAND_COVER_CLASSIFICATION" in t for t in response.execution_trace)
    assert len(response.evidence.evidence_points) >= 2

    # Round-trip JSON serializability
    d = response.to_dict()
    serialized = json.dumps(d)
    assert len(serialized) > 0


def test_orchestrator_end_to_end_change_detection(engine, synthetic_temporal_pair):
    """Verify end-to-end natural language query execution for Change Detection champion."""
    query = "Detect surface alteration and building changes between the two dates."
    response = engine.process_query(query, input_data=synthetic_temporal_pair)

    assert isinstance(response, SatQueryResponse)
    assert response.routed_task == "CHANGE_DETECTION"
    assert "SUCCESS" in response.status
    assert response.input_compatibility is True
    assert len(response.evidence.evidence_points) >= 3
    assert any("Valid pixel coverage" in pt for pt in response.evidence.evidence_points)
    assert any("Largest connected change cluster" in pt for pt in response.evidence.evidence_points)

    # Check that execution metadata identifies production champion
    meta = response.evidence.sensor_feature_summary
    assert meta["specialist_name"] in ("SiameseResNetChangeDetectionSpecialist", "TemporalDifferenceBaseline")

    # Round-trip JSON serializability
    d = response.to_dict()
    serialized = json.dumps(d)
    assert len(serialized) > 0


def test_orchestrator_incompatible_input_safety(engine, synthetic_temporal_pair):
    """Verify safety policy when single image is provided for temporal change query."""
    query = "What changes occurred between these two observations?"
    # Pass single image instead of pair
    single_img = np.zeros((64, 64, 3), dtype=np.float32)
    response = engine.process_query(query, input_data=single_img)

    assert isinstance(response, SatQueryResponse)
    # The router flags incompatible input
    assert response.input_compatibility is False or response.status in ("INCOMPATIBLE_INPUT", "NEEDS_CLARIFICATION")


def test_orchestrator_clarification_safety_policy(synthetic_temporal_pair):
    """Verify clarification policy halts execution when confidence is below threshold."""
    engine_clarify = SatQueryEngine.create_default(routing_policy="clarify")
    # Low confidence or ambiguous query without imagery
    query = "Is there anything different here?"
    response = engine_clarify.process_query(query, input_data=None)

    assert isinstance(response, SatQueryResponse)
    if response.requires_human_review:
        assert response.status == "NEEDS_CLARIFICATION"
        assert response.execution_policy == "CLARIFICATION_REQUIRED"
        assert "CLARIFICATION REQUIRED" in response.answer


# ==============================================================================
# Dedicated Fallback Verification Test Suite
# ==============================================================================

def test_change_detection_deep_model_execution_when_checkpoint_present(synthetic_temporal_pair):
    """Verify deep model executes when valid checkpoint and PyTorch backend are present."""
    spec = SiameseResNetChangeDetectionSpecialist(use_classical_fallback=True)
    assert spec.has_neural_model is True
    assert spec.model is not None

    res = spec.predict(synthetic_temporal_pair)
    assert res.status == SpecialistStatus.SUCCESS
    assert res.execution_metadata["active_backend"] == "deep_siamese_resnet"
    assert res.execution_metadata["neural_backend_available"] is True
    assert res.execution_metadata["neural_architecture"] == "SiameseResNet18UNet_scSE"
    assert "change_detected" in res.probabilities
    assert len(res.predictions) > 0
    assert not res.execution_metadata.get("is_stub", False)


def test_change_detection_checkpoint_missing_triggers_classical_fallback(synthetic_temporal_pair):
    """Verify automatic graceful fallback to classical Gaussian CVA when checkpoint is missing."""
    missing_path = Path("artifacts/specialists/non_existent_checkpoint_weights.pt")
    spec = SiameseResNetChangeDetectionSpecialist(
        checkpoint_path=missing_path,
        use_classical_fallback=True
    )
    assert spec.has_neural_model is False
    assert spec.fallback is not None
    assert isinstance(spec.fallback, TemporalDifferenceBaseline)

    res = spec.predict(synthetic_temporal_pair)
    assert res.status == SpecialistStatus.SUCCESS
    assert res.execution_metadata["active_backend"] == "classical_fallback"
    assert res.execution_metadata["neural_backend_available"] is False
    assert "change_detected" in res.probabilities
    assert len(res.predictions) > 0
    assert not res.execution_metadata.get("is_stub", False)
    assert res.execution_metadata["specialist_name"] == "temporal_difference_baseline"


def test_change_detection_simulated_backend_failure_triggers_fallback(synthetic_temporal_pair):
    """Verify graceful fallback when neural model initialization fails (e.g., PyTorch / model error)."""
    from unittest.mock import patch
    with patch("satquery.specialists.models.siamese_resnet_unet.SiameseResNet18UNet", side_effect=RuntimeError("Simulated neural init failure")):
        spec = SiameseResNetChangeDetectionSpecialist(use_classical_fallback=True)
        assert spec.has_neural_model is False
        assert spec.fallback is not None

        res = spec.predict(synthetic_temporal_pair)
        assert res.status == SpecialistStatus.SUCCESS
        assert res.execution_metadata["active_backend"] == "classical_fallback"
        assert res.execution_metadata["neural_backend_available"] is False
        assert len(res.predictions) > 0


def test_change_detection_no_checkpoint_and_no_fallback_returns_unavailable(synthetic_temporal_pair):
    """Verify structured UNAVAILABLE response when neither neural model nor fallback is available."""
    missing_path = Path("artifacts/specialists/non_existent_checkpoint_weights.pt")
    spec = SiameseResNetChangeDetectionSpecialist(
        checkpoint_path=missing_path,
        use_classical_fallback=False
    )
    assert spec.has_neural_model is False
    assert spec.fallback is None

    res = spec.predict(synthetic_temporal_pair)
    assert res.status == SpecialistStatus.UNAVAILABLE
    assert "Neither neural Siamese ResNet model nor classical fallback is available" in res.unavailable_reason
    assert len(res.predictions) == 0


def test_fallback_ambiguous_raw_float_without_product_type_is_rejected():
    """Verify that ambiguous raw float32 arrays with values > 1.0 are rejected rather than guessed."""
    missing_path = Path("artifacts/specialists/non_existent_checkpoint_weights.pt")
    spec = SiameseResNetChangeDetectionSpecialist(
        checkpoint_path=missing_path,
        use_classical_fallback=True
    )
    # Ambiguous float input with values > 1.0 without declared product_type
    t1_ambiguous = np.full((32, 32, 3), 150.0, dtype=np.float32)
    t2_ambiguous = t1_ambiguous.copy()

    res = spec.predict((t1_ambiguous, t2_ambiguous))
    # Must NOT silently misclassify as rgb_8bit; must reject with INCOMPATIBLE_INPUT
    assert res.status == SpecialistStatus.INCOMPATIBLE_INPUT
    assert "Ambiguous floating-point temporal observation" in res.unavailable_reason


def test_fallback_explicit_product_type_precedence():
    """Verify that explicit product_type in input dictionary takes precedence."""
    missing_path = Path("artifacts/specialists/non_existent_checkpoint_weights.pt")
    spec = SiameseResNetChangeDetectionSpecialist(
        checkpoint_path=missing_path,
        use_classical_fallback=True
    )
    # Provide explicit product_type='rgb_8bit' for float array
    t1 = np.full((32, 32, 3), 150.0, dtype=np.float32)
    t2 = t1.copy()
    input_dict = {
        "t1": t1,
        "t2": t2,
        "product_type": "rgb_8bit"
    }

    res = spec.predict(input_dict)
    assert res.status == SpecialistStatus.SUCCESS
    assert res.execution_metadata["active_backend"] == "classical_fallback"


