"""
SatQuery AI: Interactive Vision-Language Assistant for Multimodal Remote-Sensing Image Analysis.

Based on SIH 2026 Problem Statement SIH26167 (ISRO).
"""

__version__ = "0.1.0"

from satquery.taxonomy import TaskType, InputType, TASK_REGISTRY
from satquery.router import TaskRouter, ExecutionRoute
from satquery.specialists import BaseSpecialist, SpecialistResult, SpecialistStatus, SpecialistRegistry, LandCoverClassifier
from satquery.evidence import format_evidence, EvidenceReport
from satquery.orchestrator import SatQueryEngine, SatQueryResponse

__all__ = [
    "__version__",
    "TaskType",
    "InputType",
    "TASK_REGISTRY",
    "TaskRouter",
    "ExecutionRoute",
    "BaseSpecialist",
    "SpecialistResult",
    "SpecialistStatus",
    "SpecialistRegistry",
    "LandCoverClassifier",
    "format_evidence",
    "EvidenceReport",
    "SatQueryEngine",
    "SatQueryResponse"
]
