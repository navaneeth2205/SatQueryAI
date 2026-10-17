"""
Evidence and Prediction Formatter for SatQuery AI.

Transforms raw specialist outputs into interpretable natural-language answers
and statistical evidence summaries without inventing uncalibrated confidence metrics.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
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

    # 2. Format Successful Predictions (Land Cover)
    detected = result.predictions
    probabilities = result.probabilities
    thresholds = result.thresholds_applied
    meta = result.execution_metadata

    evidence_points = []

    if detected:
        answer = f"Detected land-cover classes: {', '.join(detected)}."
        for cls in detected:
            score = probabilities.get(cls, 0.0)
            thresh = thresholds.get(cls, 0.40)
            evidence_points.append(
                f"[Detection] '{cls}' confirmed (model vote score: {score:.2f}, adaptive threshold: {thresh:.2f})"
            )
    else:
        answer = "No registered land-cover classes exceeded their detection thresholds for this scene."
        evidence_points.append("[Detection] All class scores remained below their decision thresholds.")

    # Add verified sensor statistical evidence
    s1_summary = meta.get("radar_s1_features_summary")
    if s1_summary:
        evidence_points.append(
            f"[Radar Evidence] Sentinel-1 SAR mean backscatter: Band 1 = {s1_summary.get('band_1_mean_backscatter_db')} dB, Band 2 = {s1_summary.get('band_2_mean_backscatter_db')} dB"
        )

    s2_summary = meta.get("optical_s2_features_summary")
    if s2_summary:
        evidence_points.append(
            f"[Optical Evidence] Sentinel-2 multispectral: {s2_summary.get('bands_processed')} channels processed with mean reflectance {s2_summary.get('overall_reflectance_mean')}"
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
