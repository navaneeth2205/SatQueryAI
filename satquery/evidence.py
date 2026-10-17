"""
Evidence and Prediction Formatter for SatQuery AI.

Transforms raw specialist outputs into interpretable natural-language answers
and statistical evidence summaries without inventing uncalibrated confidence metrics.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from satquery.taxonomy import TaskType
from satquery.specialists.base import SpecialistResult, SpecialistStatus


@dataclass
class EvidenceReport:
    """Formatted evidence and natural-language interpretation."""
    answer: str
    status: str
    detected_classes: List[str]
    class_probabilities: Dict[str, float]
    thresholds_applied: Dict[str, float]
    evidence_points: List[str]
    sensor_feature_summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def format_evidence(result: SpecialistResult) -> EvidenceReport:
    """
    Format a specialist output into a transparent, explainable evidence report.
    """
    # 1. Handle Incompatible Inputs or Unavailable Stubs
    if result.status != SpecialistStatus.SUCCESS:
        answer = f"Task [{result.task}] cannot be executed: {result.unavailable_reason}"
        evidence_points = [
            f"[Status] {result.status.value}",
            f"[Reason] {result.unavailable_reason}"
        ]
        if result.required_inputs_description:
            evidence_points.append(f"[Input Requirement] {result.required_inputs_description}")

        return EvidenceReport(
            answer=answer,
            status=result.status.value,
            detected_classes=[],
            class_probabilities={},
            thresholds_applied={},
            evidence_points=evidence_points,
            sensor_feature_summary=result.execution_metadata
        )

    meta = result.execution_metadata

    # 2. Format Change Detection Specialist Output
    if result.task == TaskType.CHANGE_DETECTION.value:
        detected = result.predictions
        probabilities = result.probabilities
        thresholds = result.thresholds_applied
        changed_pct = meta.get("changed_area_pct_of_valid", 0.0)
        valid_pixels = meta.get("valid_pixels", 0)
        total_pixels = meta.get("total_pixels", 0)
        valid_pct = meta.get("valid_pixel_pct", 100.0)
        tau = thresholds.get("decision_threshold", 0.28)
        deltas = meta.get("mean_channel_deltas", [])
        bbox = meta.get("largest_change_cluster_bbox_pixel")
        cluster_area = meta.get("largest_cluster_pixel_area", 0)

        evidence_points = [
            f"[Change Prediction] Predicted outcome: '{detected[0]}' (valid-pixel change area: {changed_pct:.2f}%; decision threshold: {tau:.2f})",
            f"[Data Quality] Valid pixel coverage: {valid_pct:.1f}% ({valid_pixels}/{total_pixels} pixels; NoData/finite-value masking applied)",
            f"[Sensor Differential] Mean channel deltas across valid pixels: {deltas} (normalized absolute difference per channel, uncalibrated digital values)"
        ]

        if bbox is not None:
            evidence_points.append(
                f"[Spatial Focus] Largest connected change cluster: {cluster_area} px, bounding box = {bbox} (unprojected pixel coordinates [ymin, xmin, ymax, xmax])"
            )
        else:
            evidence_points.append(
                "[Spatial Focus] Largest connected change cluster: null (no connected changed regions >= minimum area threshold)"
            )

        if "Surface alteration detected" in detected:
            answer = f"Predicted surface change: Alteration detected across {changed_pct:.2f}% of valid scene pixels (decision threshold {tau:.2f})."
        else:
            answer = f"Predicted surface change: No significant change detected ({changed_pct:.2f}% changed pixels, below significance threshold)."

        return EvidenceReport(
            answer=answer,
            status=result.status.value,
            detected_classes=detected,
            class_probabilities=probabilities,
            thresholds_applied=thresholds,
            evidence_points=evidence_points,
            sensor_feature_summary=meta
        )

    # 3. Format Land Cover Classification Output
    detected = result.predictions
    probabilities = result.probabilities
    thresholds = result.thresholds_applied

    evidence_points = []

    if detected:
        answer = f"Predicted land-cover classes: {', '.join(detected)}."
        for cls in detected:
            score = probabilities.get(cls, 0.0)
            thresh = thresholds.get(cls, 0.40)
            evidence_points.append(
                f"[Prediction] Predicted present: '{cls}' (model score: {score:.2f}; decision threshold: {thresh:.2f})"
            )
    else:
        answer = "No registered land-cover classes exceeded their decision thresholds for this scene."
        evidence_points.append("[Prediction] All class scores remained below their decision thresholds.")

    # Add verified sensor statistical evidence
    s1_summary = meta.get("radar_s1_features_summary")
    if s1_summary:
        note = s1_summary.get("unit_note", "uncalibrated")
        evidence_points.append(
            f"[Radar Evidence] Sentinel-1 stored band statistics: Band 1 mean = {s1_summary.get('band_1_mean_stored_value')}, Band 2 mean = {s1_summary.get('band_2_mean_stored_value')} ({note})"
        )

    s2_summary = meta.get("optical_s2_features_summary")
    if s2_summary:
        note = s2_summary.get("unit_note", "uncalibrated")
        evidence_points.append(
            f"[Optical Evidence] Sentinel-2 stored band statistics: {s2_summary.get('channels_processed', 10)} channels processed with overall mean digital value = {s2_summary.get('overall_mean_stored_dn')} ({note})"
        )

    return EvidenceReport(
        answer=answer,
        status=result.status.value,
        detected_classes=detected,
        class_probabilities=probabilities,
        thresholds_applied=thresholds,
        evidence_points=evidence_points,
        sensor_feature_summary=meta
    )
