"""
Tests for Task and Modality Taxonomy.
"""

from satquery.taxonomy import (
    TaskType,
    InputType,
    TASK_REGISTRY,
    get_task_definition
)


def test_task_types_count():
    assert len(TaskType) == 7
    for task in TaskType:
        assert task in TASK_REGISTRY


def test_task_metadata_integrity():
    for task, definition in TASK_REGISTRY.items():
        assert definition.task == task
        assert len(definition.display_name) > 0
        assert len(definition.description) > 0
        assert isinstance(definition.required_input, InputType)
        assert len(definition.specialist_model_family) > 0
        assert len(definition.primary_benchmark_datasets) > 0
        assert len(definition.evaluation_metrics) > 0
        assert len(definition.default_evidence_type) > 0


def test_get_task_definition():
    lc_def = get_task_definition("LAND_COVER_CLASSIFICATION")
    assert lc_def is not None
    assert lc_def.task == TaskType.LAND_COVER_CLASSIFICATION
    assert lc_def.required_input == InputType.SINGLE_IMAGE

    cd_def = get_task_definition("CHANGE_DETECTION")
    assert cd_def is not None
    assert cd_def.required_input == InputType.TEMPORAL_PAIR

    invalid = get_task_definition("NON_EXISTENT_TASK")
    assert invalid is None
