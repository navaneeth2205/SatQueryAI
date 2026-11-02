"""
Tests for M3.0 Clarification and Routing Safety Invariants.
Directly verifies the 5 critical research reviewer requirements:
1. Ambiguous query + exactly one compatible candidate -> clarification (halted, no dispatch).
2. Ambiguous query + multiple compatible candidates -> clarification (halted, no dispatch).
3. Ambiguous query + no compatible candidates -> clarification (halted, no dispatch).
4. Clear intent + incompatible observation -> rejection (halted, no alternative dispatch).
5. Clear intent + compatible observation + low selective confidence -> abstention / review flag.
"""

import numpy as np
import pytest
from pathlib import Path

from satquery.orchestrator import SatQueryEngine, SatQueryResponse
from satquery.router import TaskRouter, ExecutionRoute
from satquery.taxonomy import TaskType, InputType
from satquery.utils.metadata import InputObservationMetadata


@pytest.fixture
def engine():
    """Instantiate standard production orchestrator under clarify policy."""
    return SatQueryEngine.create_default(routing_policy="clarify")


@pytest.fixture
def router():
    """Load trained routing model."""
    repo_root = Path(__file__).resolve().parent.parent
    router_path = repo_root / "artifacts" / "router" / "router_model.joblib"
    return TaskRouter.load(router_path)


# 1. Ambiguous query + exactly one compatible candidate -> clarification
def test_ambiguous_query_single_compatible_candidate_triggers_clarification(engine):
    # Query is broad and uncommitted (ambiguous intent, router confidence < 0.65)
    query = "Analyze and survey this satellite observation."
    # Attached input is a temporal pair (only Change Detection / Change VQA compatible)
    temp_meta = InputObservationMetadata(
        modality=InputType.TEMPORAL_PAIR.value,
        image_count=2,
        has_temporal_context=True,
        is_valid_temporal_pair=True
    )
    response = engine.process_query(query, input_data=temp_meta)

    # Invariant: Observation compatibility MUST NEVER automatically dispatch an ambiguous query
    assert response.status == "NEEDS_CLARIFICATION"
    assert response.execution_policy == "CLARIFICATION_REQUIRED"
    assert response.requires_human_review is True
    assert "CLARIFICATION REQUIRED" in response.answer


# 2. Ambiguous query + multiple compatible candidates -> clarification
def test_ambiguous_query_multiple_compatible_candidates_triggers_clarification(engine):
    query = "Please assess this region."
    # Single image compatible with VQA, Captioning, Visual Grounding
    single_meta = InputObservationMetadata(
        modality=InputType.SINGLE_IMAGE.value,
        image_count=1,
        sensors=["optical"]
    )
    response = engine.process_query(query, input_data=single_meta)

    assert response.status == "NEEDS_CLARIFICATION"
    assert response.execution_policy == "CLARIFICATION_REQUIRED"
    assert response.requires_human_review is True
    assert "CLARIFICATION REQUIRED" in response.answer


# 3. Ambiguous query + no compatible candidates -> clarification
def test_ambiguous_query_no_compatible_candidates_triggers_clarification(engine):
    query = "Inspect this target."
    # Unknown / empty / invalid metadata where 0 specialists are compatible
    invalid_meta = InputObservationMetadata(
        modality="corrupted_unknown",
        image_count=0
    )
    response = engine.process_query(query, input_data=invalid_meta)

    # Ambiguity takes precedence: must clarify rather than fail on input
    assert response.status == "NEEDS_CLARIFICATION"
    assert response.execution_policy == "CLARIFICATION_REQUIRED"
    assert response.requires_human_review is True


# 4. Clear intent + incompatible observation -> rejection, no alternative dispatch
def test_clear_intent_incompatible_observation_rejects_without_coercion(engine):
    # Clear Land Cover intent (confidence >= 0.65)
    query = "Classify the land cover and crop types across this agricultural region."
    # Attach raw single image (incompatible with Land Cover which requires S1/S2 pair or features)
    raw_single_img = np.zeros((120, 120, 3), dtype=np.uint8)
    
    response = engine.process_query(query, input_data=raw_single_img)

    # Invariant: Must halt with INCOMPATIBLE_INPUT and must NOT dispatch any alternative specialist
    assert response.routed_task == "LAND_COVER_CLASSIFICATION"
    assert response.status == "INCOMPATIBLE_INPUT"
    assert response.execution_policy == "STANDARD_EXECUTION"
    # Never coerced to Change Detection, Single Image VQA, or Scene Captioning
    assert response.routed_task != "CHANGE_DETECTION"
    assert response.routed_task != "SINGLE_IMAGE_VQA"
    assert "incompatible" in response.answer.lower() or "requires" in response.answer.lower()


# 5. Clear intent + compatible observation + low selective confidence -> review flag / abstention
def test_clear_intent_compatible_observation_low_selective_confidence_flags_review(router):
    # Query with moderate confidence near threshold
    query = "Examine the fields and waterways."
    # Test router with elevated selective threshold (e.g. tau_sel = 0.95)
    strict_router = TaskRouter(confidence_threshold=0.95)
    strict_router.pipeline = router.pipeline
    strict_router.classes_ = router.classes_
    strict_router.is_fitted = True

    route = strict_router.predict(query)
    # When confidence is below selective threshold, requires_review must be True (abstention flag)
    assert route.confidence < 0.95
    assert route.requires_review is True
