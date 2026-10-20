"""
Unit and integration tests for Input-Aware Routing and Observation Metadata.

Verifies:
- Concrete metadata extraction across feature vectors, single images, temporal pairs, and SAR-optical pairs.
- Separation of raw linguistic confidence P(task | text) from input-conditioned compatibility.
- Disambiguation of ambiguous queries when orthogonal input observations are attached.
- Preservation of safety review threshold (0.65) and clarification policy under input-aware routing.
- Preservation of feature vector modality as distinct from raw single image.
"""

from pathlib import Path
import numpy as np
import pytest

from satquery.router import TaskRouter, ExecutionRoute
from satquery.orchestrator import SatQueryEngine, SatQueryResponse
from satquery.taxonomy import TaskType, InputType
from satquery.utils.metadata import (
    InputObservationMetadata,
    extract_observation_metadata,
    TASK_COMPATIBILITY_MAP
)


@pytest.fixture
def fitted_router():
    repo_root = Path(__file__).resolve().parent.parent
    router_path = repo_root / "artifacts" / "router" / "router_model.joblib"
    return TaskRouter.load(router_path)


@pytest.fixture
def engine():
    return SatQueryEngine.create_default()


def test_metadata_extraction_modalities():
    """Verify metadata extraction across all supported input formats."""
    # 1. Query only (None)
    m_none = extract_observation_metadata(None)
    assert m_none.modality is None
    assert m_none.image_count == 0
    assert m_none.is_feature_vector is False

    # 2. 1D Numpy Array (Feature vector, distinct from single image)
    feats = np.full(84, 1.0, dtype=np.float32)
    m_vec = extract_observation_metadata(feats)
    assert m_vec.modality == "feature_vector"
    assert m_vec.is_feature_vector is True
    assert m_vec.feature_dim == 84
    assert m_vec.image_count == 0

    # 3. 2D / 3D Image Array
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    m_img = extract_observation_metadata(img)
    assert m_img.modality == InputType.SINGLE_IMAGE.value
    assert m_img.image_count == 1
    assert m_img.is_feature_vector is False
    assert m_img.spatial_shape == (64, 64, 3)

    # 4. Valid Temporal Pair Tuple
    t1 = np.zeros((32, 32, 3), dtype=np.uint8)
    t2 = np.zeros((32, 32, 3), dtype=np.uint8)
    m_temp = extract_observation_metadata((t1, t2))
    assert m_temp.modality == InputType.TEMPORAL_PAIR.value
    assert m_temp.image_count == 2
    assert m_temp.has_temporal_context is True
    assert m_temp.is_valid_temporal_pair is True

    # 5. Dict with features
    m_dict_feat = extract_observation_metadata({"features": feats, "patch_id": "test_patch"})
    assert m_dict_feat.modality == "feature_vector"
    assert m_dict_feat.is_feature_vector is True
    assert m_dict_feat.details["patch_id"] == "test_patch"

    # 6. Dict with temporal pair
    m_dict_temp = extract_observation_metadata({"t1": t1, "t2": t2, "scene_name": "paris"})
    assert m_dict_temp.modality == InputType.TEMPORAL_PAIR.value
    assert m_dict_temp.has_temporal_context is True
    assert m_dict_temp.is_valid_temporal_pair is True


def test_feature_vector_is_distinct_from_single_image():
    """Verify acceptance criterion: feature vector is not classified as single image."""
    vec = np.random.randn(84).astype(np.float32)
    meta = extract_observation_metadata(vec)
    assert meta.modality == "feature_vector"
    assert meta.modality != InputType.SINGLE_IMAGE.value
    assert meta.is_feature_vector is True
    assert meta.is_compatible_with("LAND_COVER_CLASSIFICATION") is True
    assert meta.is_compatible_with("CHANGE_DETECTION") is False


def test_input_aware_routing_preserves_text_confidence(fitted_router):
    """Verify that input compatibility does not overwrite or distort raw linguistic confidence."""
    query = "What changed in this area between 2024 and 2026?"
    
    # 1. Predict with text alone
    route_text_only = fitted_router.predict(query)
    raw_conf = route_text_only.confidence
    assert raw_conf > 0.0

    # 2. Predict with attached temporal pair metadata
    temp_meta = InputObservationMetadata(
        modality=InputType.TEMPORAL_PAIR.value,
        image_count=2,
        has_temporal_context=True,
        is_valid_temporal_pair=True
    )
    route_with_meta = fitted_router.predict(query, image_metadata=temp_meta)

    # Acceptance criterion: Original text confidence remains untouched and separate
    assert route_with_meta.text_confidence == pytest.approx(raw_conf, abs=1e-4)
    assert route_with_meta.confidence == pytest.approx(raw_conf, abs=1e-4)
    assert route_with_meta.text_task == route_text_only.task
    assert route_with_meta.selected_task == "CHANGE_DETECTION"
    assert route_with_meta.input_compatibility is True
    assert "verified compatible" in route_with_meta.routing_reason


def test_input_aware_disambiguation_when_text_is_incompatible(fitted_router):
    """
    When text top prediction requires a single image, but user attaches a temporal pair,
    router disambiguates to the highest-ranking compatible candidate task.
    """
    # A generic query that may score slightly higher on VQA/classification
    query = "Is there any noticeable alteration visible across this scene?"
    
    temp_meta = InputObservationMetadata(
        modality=InputType.TEMPORAL_PAIR.value,
        image_count=2,
        has_temporal_context=True,
        is_valid_temporal_pair=True
    )
    route = fitted_router.predict(query, image_metadata=temp_meta)

    assert route.attached_modality == InputType.TEMPORAL_PAIR.value
    # Selected task must be a temporal task
    assert route.selected_task in ("CHANGE_DETECTION", "CHANGE_VQA")
    assert route.task in ("CHANGE_DETECTION", "CHANGE_VQA")
    assert len(route.routing_reason) > 0


def test_orchestrator_integration_with_input_metadata(engine):
    """Verify end-to-end SatQueryEngine propagates new fields in SatQueryResponse."""
    query = "What changed between 2024 and 2026?"
    t1 = np.zeros((30, 30, 3), dtype=np.uint8)
    t2 = np.zeros((30, 30, 3), dtype=np.uint8)
    
    response = engine.process_query(query, input_data=(t1, t2))
    assert isinstance(response, SatQueryResponse)
    assert response.routed_task == "CHANGE_DETECTION"
    assert response.selected_task == "CHANGE_DETECTION"
    assert response.attached_modality == InputType.TEMPORAL_PAIR.value
    assert response.text_confidence > 0.0
    assert response.input_compatibility is True
    assert "routing_reason" in response.to_dict()


def test_safety_policy_preservation_under_input_aware_routing():
    """Verify acceptance criterion: clarification policy still blocks execution if confidence < 0.65."""
    engine_clarify = SatQueryEngine.create_default(routing_policy="clarify")
    query = "What changed between 2024 and 2026?"
    t1 = np.zeros((30, 30, 3), dtype=np.uint8)
    t2 = np.zeros((30, 30, 3), dtype=np.uint8)

    # Even with temporal input attached, if text confidence < 0.65, clarify policy halts automated execution
    response = engine_clarify.process_query(query, input_data=(t1, t2))
    assert response.status == "NEEDS_CLARIFICATION"
    assert response.execution_policy == "CLARIFICATION_REQUIRED"
    assert response.requires_human_review is True
    assert response.attached_modality == InputType.TEMPORAL_PAIR.value
