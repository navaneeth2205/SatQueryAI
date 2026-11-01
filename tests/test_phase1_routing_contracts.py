"""
Phase 1 Regression Tests: Router & Compatibility Engine Hardening.

Verifies:
1. Anti-coercion: Temporal pair attached to Land Cover classification preserves intended task and sets input_compatibility=False.
2. Anti-coercion: Single image attached to Change Detection preserves intended task and sets input_compatibility=False.
3. Disambiguation: Ambiguous alteration query with temporal pair safely disambiguates to Change Detection with review flag.
4. Disambiguation: Broad uncommitted query with feature vector safely disambiguates to Land Cover with review flag.
5. Contract enforcement: Raw single image rejected by Land Cover contract (requires feature vector or S1/S2 pair).
6. Contract enforcement: Multimodal S1/S2 GeoTIFF pair and feature vector accepted by Land Cover contract.
7. Contract registration: Formal SpecialistInputContract registered on all 7 taxonomy tasks.
8. Roadmap stubs: Unregistered vision-language tasks marked with is_registered=False.
9. Orchestrator policy: Status and execution policies map strictly to standardized action taxonomy.
10. Operating threshold consistency: Increasing confidence threshold monotonically controls selective risk.
"""

from pathlib import Path
import numpy as np
import pytest

from satquery.taxonomy import TaskType, InputType, TASK_REGISTRY, get_task_definition, get_task_contract, SpecialistInputContract
from satquery.router import TaskRouter, ExecutionRoute
from satquery.orchestrator import SatQueryEngine, SatQueryResponse
from satquery.utils.metadata import (
    InputObservationMetadata,
    extract_observation_metadata,
    TASK_COMPATIBILITY_MAP
)


@pytest.fixture
def router():
    repo_root = Path(__file__).resolve().parent.parent
    router_path = repo_root / "artifacts" / "router" / "router_model.joblib"
    return TaskRouter.load(router_path)


@pytest.fixture
def engine():
    return SatQueryEngine.create_default(routing_policy="clarify")


# Test 1: Task Coercion Prevention (Land Cover + Temporal Pair)
def test_task_coercion_prevention_land_cover_with_temporal_pair(router):
    query = "What land-cover classes are present in this satellite scene?"
    temp_meta = InputObservationMetadata(
        modality=InputType.TEMPORAL_PAIR.value,
        image_count=2,
        has_temporal_context=True,
        is_valid_temporal_pair=True
    )
    route = router.predict(query, image_metadata=temp_meta)

    # Intended task MUST NOT be coerced to CHANGE_DETECTION
    assert route.text_task == "LAND_COVER_CLASSIFICATION"
    assert route.selected_task == "LAND_COVER_CLASSIFICATION"
    assert route.task == "LAND_COVER_CLASSIFICATION"
    assert route.input_compatibility is False
    assert "prevent task coercion" in route.routing_reason


# Test 2: Task Coercion Prevention (Change Detection + Single Image)
def test_task_coercion_prevention_change_detection_with_single_image(router):
    query = "What changed in this area between 2024 and 2026?"
    single_meta = InputObservationMetadata(
        modality=InputType.SINGLE_IMAGE.value,
        image_count=1,
        sensors=["optical"]
    )
    route = router.predict(query, image_metadata=single_meta)

    # Intended task MUST NOT be coerced to single-image task
    assert route.text_task == "CHANGE_DETECTION"
    assert route.selected_task == "CHANGE_DETECTION"
    assert route.task == "CHANGE_DETECTION"
    assert route.input_compatibility is False
    assert "prevent task coercion" in route.routing_reason


# Test 3: Valid Disambiguation for Ambiguous Alteration Query
def test_valid_disambiguation_for_ambiguous_alteration_query(router):
    query = "Is there any noticeable alteration visible across this scene?"
    temp_meta = InputObservationMetadata(
        modality=InputType.TEMPORAL_PAIR.value,
        image_count=2,
        has_temporal_context=True,
        is_valid_temporal_pair=True
    )
    route = router.predict(query, image_metadata=temp_meta)

    assert route.attached_modality == InputType.TEMPORAL_PAIR.value
    assert route.selected_task in ("CHANGE_DETECTION", "CHANGE_VQA")
    assert route.input_compatibility is True
    assert route.requires_review is True  # Ambiguity review flag preserved


# Test 4: Valid Disambiguation for Broad Query with Feature Vector
def test_valid_disambiguation_for_broad_query_with_feature_vector(router):
    query = "Analyze this satellite observation and report findings."
    vec_meta = InputObservationMetadata(
        modality="feature_vector",
        image_count=0,
        is_feature_vector=True,
        feature_dim=488
    )
    route = router.predict(query, image_metadata=vec_meta)

    assert route.attached_modality == "feature_vector"
    assert route.selected_task == "LAND_COVER_CLASSIFICATION"
    assert route.input_compatibility is True
    assert route.requires_review is True


# Test 5: Raw Single Image Rejected by Land Cover Contract
def test_raw_single_image_rejected_by_land_cover_contract():
    raw_img = np.zeros((256, 256, 3), dtype=np.uint8)
    meta = extract_observation_metadata(raw_img)

    assert meta.modality == InputType.SINGLE_IMAGE.value
    # Land Cover specialist is tabular (requires 84/488 features or S1+S2 GeoTIFFs),
    # so raw 2D/3D images must be strictly incompatible.
    assert meta.is_compatible_with("LAND_COVER_CLASSIFICATION") is False


# Test 6: Multimodal S1/S2 GeoTIFF Pair and Features Accepted by Land Cover
def test_multimodal_s1_s2_and_features_accepted_by_land_cover():
    # Feature vector
    feats = np.ones(488, dtype=np.float32)
    meta_vec = extract_observation_metadata(feats)
    assert meta_vec.is_compatible_with("LAND_COVER_CLASSIFICATION") is True

    # Optical-SAR GeoTIFF pair
    meta_sar = InputObservationMetadata(
        modality=InputType.OPTICAL_SAR_PAIR.value,
        image_count=2,
        sensors=["sar", "optical"]
    )
    assert meta_sar.is_compatible_with("LAND_COVER_CLASSIFICATION") is True


# Test 7: Specialist Input Contracts Registered on All 7 Taxonomy Tasks
def test_specialist_input_contracts_registered_on_all_tasks():
    assert len(TaskType) == 7
    for task in TaskType:
        t_def = get_task_definition(task.value)
        assert t_def is not None
        assert t_def.contract is not None
        contract = t_def.contract
        assert isinstance(contract, SpecialistInputContract)
        assert contract.task == task
        assert len(contract.required_modalities) > 0
        assert len(contract.accepted_representations) > 0
        assert contract.temporal_relationship in ("single_point", "bitemporal_pair")
        assert len(contract.accepted_dtypes) > 0
        assert len(contract.accepted_scalings) > 0


# Test 8: Roadmap Stubs Marked with is_registered=False
def test_roadmap_stubs_marked_with_is_registered_false():
    # Deployed specialists
    assert get_task_contract("LAND_COVER_CLASSIFICATION").is_registered is True
    assert get_task_contract("CHANGE_DETECTION").is_registered is True

    # Unregistered stubs
    assert get_task_contract("SINGLE_IMAGE_VQA").is_registered is False
    assert get_task_contract("SCENE_CAPTIONING").is_registered is False
    assert get_task_contract("VISUAL_GROUNDING").is_registered is False
    assert get_task_contract("CHANGE_VQA").is_registered is False
    assert get_task_contract("OPTICAL_SAR_FUSION").is_registered is False


# Test 9: Standardized Action Taxonomy in Orchestrator Responses
def test_standardized_action_taxonomy_in_orchestrator(engine):
    # 1. Compatible Land Cover Request -> SUCCESS
    lc_query = "What land-cover classes are present in this satellite scene?"
    res_success = engine.process_query(lc_query, input_data=np.zeros(488, dtype=np.float32))
    assert res_success.status == "SUCCESS"
    assert res_success.execution_policy == "STANDARD_EXECUTION"

    # 2. Incompatible Land Cover Request with Temporal Pair -> INCOMPATIBLE_INPUT
    t1 = np.zeros((64, 64, 3), dtype=np.uint8)
    t2 = np.zeros((64, 64, 3), dtype=np.uint8)
    res_incompat = engine.process_query(lc_query, input_data=(t1, t2))
    assert res_incompat.status == "INCOMPATIBLE_INPUT"

    # 3. Uncertain OOD Query with Clarify Policy -> NEEDS_CLARIFICATION
    ood_query = "Write a python script to compute Fibonacci numbers."
    res_clarify = engine.process_query(ood_query, input_data=None)
    assert res_clarify.status == "NEEDS_CLARIFICATION"
    assert res_clarify.execution_policy == "CLARIFICATION_REQUIRED"

    # 4. Unsupported Task Query -> UNAVAILABLE_SPECIALIST
    vqa_query = "How many commercial airplanes are parked at this airport runway?"
    res_unavail = engine.process_query(vqa_query, input_data=np.zeros((64, 64, 3), dtype=np.uint8))
    assert res_unavail.status == "UNAVAILABLE_SPECIALIST"


# Test 10: Operating Threshold Risk Control Property
def test_operating_threshold_risk_control_property(router):
    """Verify that raising threshold from 0.20 to 0.80 strictly limits execution to higher confidence."""
    router.confidence_threshold = 0.20
    # At tau=0.20, moderate confidence query passes threshold
    r_low = router.predict("What changed in this area?")
    assert r_low.requires_review is False

    # At tau=0.80, same query requires human review
    router.confidence_threshold = 0.80
    r_high = router.predict("What changed in this area?")
    assert r_high.requires_review is True

    # Reset default threshold
    router.confidence_threshold = 0.65
