"""
Specialist Models Subsystem for SatQuery AI.
"""

from satquery.specialists.base import BaseSpecialist, SpecialistResult, SpecialistStatus
from satquery.specialists.registry import SpecialistRegistry
from satquery.specialists.land_cover import LandCoverClassifier

__all__ = [
    "BaseSpecialist",
    "SpecialistResult",
    "SpecialistStatus",
    "SpecialistRegistry",
    "LandCoverClassifier"
]
