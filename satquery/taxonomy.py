"""
Task and Modality Taxonomy for SatQuery AI.

Defines the analytical categories, required input configurations,
specialist model targets, and benchmark datasets aligned with the academic research taxonomy.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class InputType(str, Enum):
    """Input configurations supported by SatQuery AI."""
    SINGLE_IMAGE = "single_image"
    TEMPORAL_PAIR = "temporal_pair"
    OPTICAL_SAR_PAIR = "optical_sar_pair"


class TaskType(str, Enum):
    """Core analytical task taxonomy."""
    LAND_COVER_CLASSIFICATION = "LAND_COVER_CLASSIFICATION"
    SINGLE_IMAGE_VQA = "SINGLE_IMAGE_VQA"
    SCENE_CAPTIONING = "SCENE_CAPTIONING"
    VISUAL_GROUNDING = "VISUAL_GROUNDING"
    CHANGE_DETECTION = "CHANGE_DETECTION"
    CHANGE_VQA = "CHANGE_VQA"
    OPTICAL_SAR_FUSION = "OPTICAL_SAR_FUSION"


@dataclass(frozen=True)
class SpecialistInputContract:
    """
    Formal execution contract defining required data modalities,
    accepted representations, temporal constraints, and numerical formats for a specialist.
    """
    task: TaskType
    required_modalities: List[str]
    accepted_representations: List[str]
    temporal_relationship: str
    accepted_dtypes: List[str]
    accepted_scalings: List[str]
    is_registered: bool = False


@dataclass(frozen=True)
class TaskDefinition:
    """Metadata and execution specification for a specialist task."""
    task: TaskType
    display_name: str
    description: str
    required_input: InputType
    specialist_model_family: str
    primary_benchmark_datasets: List[str]
    evaluation_metrics: List[str]
    default_evidence_type: str
    contract: Optional[SpecialistInputContract] = None


TASK_REGISTRY: Dict[TaskType, TaskDefinition] = {
    TaskType.LAND_COVER_CLASSIFICATION: TaskDefinition(
        task=TaskType.LAND_COVER_CLASSIFICATION,
        display_name="Multilabel Land Cover Classification",
        description="Predict presence of land-cover categories (arable, forest, water, urban) across satellite patches.",
        required_input=InputType.SINGLE_IMAGE,
        specialist_model_family="random_forest_xgboost_multilabel",
        primary_benchmark_datasets=["BigEarthNet-14k", "reBEN", "EuroSAT"],
        evaluation_metrics=["Micro F1", "Macro F1", "Hamming Loss"],
        default_evidence_type="class_probabilities_and_radar_optical_indices",
        contract=SpecialistInputContract(
            task=TaskType.LAND_COVER_CLASSIFICATION,
            required_modalities=["optical", "sar"],
            accepted_representations=["feature_vector", "multimodal_geotiff_pair"],
            temporal_relationship="single_point",
            accepted_dtypes=["float32", "float64"],
            accepted_scalings=["pre_extracted_features", "raw_digital_numbers"],
            is_registered=True
        )
    ),
    TaskType.SINGLE_IMAGE_VQA: TaskDefinition(
        task=TaskType.SINGLE_IMAGE_VQA,
        display_name="Single-Image Remote-Sensing VQA",
        description="Answer factual, counting, and presence questions about high-level features in an image.",
        required_input=InputType.SINGLE_IMAGE,
        specialist_model_family="vlm_rsvqa_specialist",
        primary_benchmark_datasets=["RSVQA", "VRSBench"],
        evaluation_metrics=["Accuracy", "Exact Match", "BLEU"],
        default_evidence_type="attention_heatmap_and_feature_crop",
        contract=SpecialistInputContract(
            task=TaskType.SINGLE_IMAGE_VQA,
            required_modalities=["optical"],
            accepted_representations=["single_image"],
            temporal_relationship="single_point",
            accepted_dtypes=["uint8", "float32"],
            accepted_scalings=["rgb_image", "s2_l2a"],
            is_registered=False
        )
    ),
    TaskType.SCENE_CAPTIONING: TaskDefinition(
        task=TaskType.SCENE_CAPTIONING,
        display_name="Scene Description and Captioning",
        description="Generate concise natural language descriptions of the geographical layout and objects.",
        required_input=InputType.SINGLE_IMAGE,
        specialist_model_family="vision_language_captioner",
        primary_benchmark_datasets=["VRSBench", "UCM-Captions"],
        evaluation_metrics=["BLEU-4", "ROUGE-L", "CIDEr", "METEOR"],
        default_evidence_type="generated_caption_tokens",
        contract=SpecialistInputContract(
            task=TaskType.SCENE_CAPTIONING,
            required_modalities=["optical"],
            accepted_representations=["single_image"],
            temporal_relationship="single_point",
            accepted_dtypes=["uint8", "float32"],
            accepted_scalings=["rgb_image"],
            is_registered=False
        )
    ),
    TaskType.VISUAL_GROUNDING: TaskDefinition(
        task=TaskType.VISUAL_GROUNDING,
        display_name="Text-Guided Visual Grounding",
        description="Localize referred objects or geographic features with bounding boxes or segmentation masks.",
        required_input=InputType.SINGLE_IMAGE,
        specialist_model_family="referring_expression_grounding",
        primary_benchmark_datasets=["VRSBench", "RSVG"],
        evaluation_metrics=["IoU", "mIoU", "mAP@0.5"],
        default_evidence_type="bounding_box_overlay_and_segmentation_mask",
        contract=SpecialistInputContract(
            task=TaskType.VISUAL_GROUNDING,
            required_modalities=["optical"],
            accepted_representations=["single_image"],
            temporal_relationship="single_point",
            accepted_dtypes=["uint8", "float32"],
            accepted_scalings=["rgb_image"],
            is_registered=False
        )
    ),
    TaskType.CHANGE_DETECTION: TaskDefinition(
        task=TaskType.CHANGE_DETECTION,
        display_name="Multitemporal Change Detection",
        description="Compare co-registered observations across time to detect geographical and infrastructural changes.",
        required_input=InputType.TEMPORAL_PAIR,
        specialist_model_family="siamese_differencing_change_detector",
        primary_benchmark_datasets=["CDVQA", "LEVIR-CD", "OSCD"],
        evaluation_metrics=["mIoU", "Precision", "Recall", "F1-Score"],
        default_evidence_type="binary_change_mask_and_difference_heatmap",
        contract=SpecialistInputContract(
            task=TaskType.CHANGE_DETECTION,
            required_modalities=["optical"],
            accepted_representations=["temporal_image_pair", "temporal_path_pair"],
            temporal_relationship="bitemporal_pair",
            accepted_dtypes=["uint8", "uint16", "float32"],
            accepted_scalings=["oscd_rgb", "s2_l2a", "pre_normalized"],
            is_registered=True
        )
    ),
    TaskType.CHANGE_VQA: TaskDefinition(
        task=TaskType.CHANGE_VQA,
        display_name="Change-Oriented Visual Question Answering",
        description="Reason about temporal alterations and answer qualitative questions about changes between dates.",
        required_input=InputType.TEMPORAL_PAIR,
        specialist_model_family="multitemporal_vqa_network",
        primary_benchmark_datasets=["CDVQA"],
        evaluation_metrics=["VQA Accuracy", "Answer Agreement"],
        default_evidence_type="temporal_differential_explanation",
        contract=SpecialistInputContract(
            task=TaskType.CHANGE_VQA,
            required_modalities=["optical"],
            accepted_representations=["temporal_image_pair"],
            temporal_relationship="bitemporal_pair",
            accepted_dtypes=["uint8", "float32"],
            accepted_scalings=["rgb_image"],
            is_registered=False
        )
    ),
    TaskType.OPTICAL_SAR_FUSION: TaskDefinition(
        task=TaskType.OPTICAL_SAR_FUSION,
        display_name="Multimodal Optical-SAR Joint Analysis",
        description="Jointly reason over complementary optical multispectral and SAR radar backscatter data.",
        required_input=InputType.OPTICAL_SAR_PAIR,
        specialist_model_family="multimodal_fusion_tammi_sar_opt",
        primary_benchmark_datasets=["BigEarthNet-S1-S2", "MM-RSVQA"],
        evaluation_metrics=["Classification F1", "Cross-Modal Consistency"],
        default_evidence_type="sar_backscatter_polarization_ratio_and_optical_indices",
        contract=SpecialistInputContract(
            task=TaskType.OPTICAL_SAR_FUSION,
            required_modalities=["optical", "sar"],
            accepted_representations=["multimodal_geotiff_pair", "feature_vector"],
            temporal_relationship="single_point",
            accepted_dtypes=["uint16", "float32"],
            accepted_scalings=["s1_sigma0", "s2_l2a"],
            is_registered=False
        )
    )
}


def get_task_definition(task_name: str) -> Optional[TaskDefinition]:
    """Retrieve task definition by enum or string name."""
    try:
        task_enum = TaskType(task_name)
        return TASK_REGISTRY.get(task_enum)
    except ValueError:
        return None


def get_task_contract(task_name_or_enum: Any) -> Optional[SpecialistInputContract]:
    """Retrieve specialist input contract by enum or string task name."""
    t_def = get_task_definition(str(task_name_or_enum))
    return t_def.contract if t_def else None
