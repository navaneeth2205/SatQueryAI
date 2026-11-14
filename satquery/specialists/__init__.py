"""
Specialist Models Subsystem for SatQuery AI.
"""

from satquery.specialists.base import BaseSpecialist, SpecialistResult, SpecialistStatus
from satquery.specialists.registry import SpecialistRegistry
from satquery.specialists.land_cover import LandCoverClassifier
from satquery.specialists.change_detection import (
    TemporalDifferenceBaseline,
    SiameseResNetChangeDetectionSpecialist,
    ChangeDetectionSpecialist,
)

__all__ = [
    "BaseSpecialist",
    "SpecialistResult",
    "SpecialistStatus",
    "SpecialistRegistry",
    "LandCoverClassifier",
    "TemporalDifferenceBaseline",
    "SiameseResNetChangeDetectionSpecialist",
    "ChangeDetectionSpecialist",
]
