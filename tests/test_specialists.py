"""
Unit and Integration Tests for Specialist Models and Registry.
"""

import numpy as np
import pytest
from satquery.taxonomy import TaskType, InputType
from satquery.specialists.base import SpecialistStatus, SpecialistResult
from satquery.specialists.registry import SpecialistRegistry
from satquery.specialists.land_cover import LandCoverClassifier


@pytest.fixture
def land_cover_specialist():
    return LandCoverClassifier()


@pytest.fixture
def dummy_84_features():
    # 14 S1 features (typical backscatter ~ -15 dB) + 70 S2 features (typical reflectance ~ 500)
    s1 = np.full(14, -15.0, dtype=np.float32)
    s2 = np.full(70, 500.0, dtype=np.float32)
    return np.concatenate([s1, s2])


def test_land_cover_specialist_initialization(land_cover_specialist):
    assert land_cover_specialist.name == "LandCoverClassifier_RF"
    assert len(land_cover_specialist.classes) == 16
    assert "Arable land" in land_cover_specialist.classes
    assert land_cover_specialist.global_threshold == 0.40
    assert land_cover_specialist.class_thresholds.get("Permanent crops") == 0.15
    assert land_cover_specialist.class_thresholds.get("Pastures") == 0.20
    assert land_cover_specialist.class_thresholds.get("Inland wetlands") == 0.35


def test_land_cover_input_validation(land_cover_specialist, dummy_84_features):
    # Valid 84 features
    is_valid, err = land_cover_specialist.supports_input(dummy_84_features)
    assert is_valid is True
    assert err is None

    # Invalid length
    is_valid, err = land_cover_specialist.supports_input(np.zeros(50))
    assert is_valid is False
    assert "Expected 84 features" in err

    # None input
    is_valid, err = land_cover_specialist.supports_input(None)
    assert is_valid is False

    # NaN in features
    nan_arr = dummy_84_features.copy()
    nan_arr[5] = np.nan
    is_valid, err = land_cover_specialist.supports_input(nan_arr)
    assert is_valid is False
    assert "NaN or infinite" in err


def test_land_cover_prediction(land_cover_specialist, dummy_84_features):
    result = land_cover_specialist.predict(dummy_84_features)
    assert isinstance(result, SpecialistResult)
    assert result.status == SpecialistStatus.SUCCESS
    assert result.task == TaskType.LAND_COVER_CLASSIFICATION.value
    assert len(result.probabilities) == 16
    assert len(result.thresholds_applied) == 16
    assert isinstance(result.predictions, list)
    assert "radar_s1_features_summary" in result.execution_metadata
    assert "optical_s2_features_summary" in result.execution_metadata


def test_specialist_registry_dispatch(land_cover_specialist, dummy_84_features):
    registry = SpecialistRegistry()
    registry.register(land_cover_specialist)

    # 1. Dispatch supported task with valid input
    res_lc = registry.dispatch(TaskType.LAND_COVER_CLASSIFICATION, dummy_84_features)
    assert res_lc.status == SpecialistStatus.SUCCESS
    assert len(res_lc.probabilities) == 16

    # 2. Dispatch supported task with invalid input
    res_incompat = registry.dispatch(TaskType.LAND_COVER_CLASSIFICATION, np.zeros(10))
    assert res_incompat.status == SpecialistStatus.INCOMPATIBLE_INPUT
    assert "Incompatible input" in res_incompat.unavailable_reason

    # 3. Dispatch unsupported task (explicit stub)
    res_cd = registry.dispatch(TaskType.CHANGE_DETECTION, None)
    assert res_cd.status == SpecialistStatus.UNAVAILABLE
    assert "No active specialist model is currently registered" in res_cd.unavailable_reason
    assert res_cd.execution_metadata.get("is_stub") is True
    assert len(res_cd.predictions) == 0  # Crucial: no fabricated outputs!

    # 4. Dispatch VQA unsupported task
    res_vqa = registry.dispatch(TaskType.SINGLE_IMAGE_VQA, None)
    assert res_vqa.status == SpecialistStatus.UNAVAILABLE
    assert "Milestone 3" in res_vqa.unavailable_reason


def test_specialist_registry_with_change_detection():
    from satquery.specialists.change_detection import TemporalDifferenceBaseline
    registry = SpecialistRegistry()
    cd_specialist = TemporalDifferenceBaseline(decision_threshold=0.25)
    registry.register(cd_specialist)

    # 1. Valid temporal pair
    t1 = np.zeros((20, 20, 3), dtype=np.float32)
    t2 = np.zeros((20, 20, 3), dtype=np.float32)
    res = registry.dispatch(TaskType.CHANGE_DETECTION, (t1, t2))
    assert res.status == SpecialistStatus.SUCCESS
    assert res.task == TaskType.CHANGE_DETECTION.value

    # 2. Incompatible input (single image)
    res_bad = registry.dispatch(TaskType.CHANGE_DETECTION, t1)
    assert res_bad.status == SpecialistStatus.INCOMPATIBLE_INPUT

    # 3. Unsupported task remains unavailable
    res_unsupported = registry.dispatch(TaskType.CHANGE_VQA, (t1, t2))
    assert res_unsupported.status == SpecialistStatus.UNAVAILABLE

