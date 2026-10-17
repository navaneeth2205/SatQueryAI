"""
Base Specialist Interface and Output Specifications for SatQuery AI.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from satquery.taxonomy import TaskType, InputType


class SpecialistStatus(str, Enum):
    """Execution status returned by a specialist."""
    SUCCESS = "SUCCESS"
    INCOMPATIBLE_INPUT = "INCOMPATIBLE_INPUT"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass
class SpecialistResult:
    """Standardized output schema across all specialist models."""
    status: SpecialistStatus
    task: str
    predictions: List[str] = field(default_factory=list)
    probabilities: Dict[str, float] = field(default_factory=dict)
    thresholds_applied: Dict[str, float] = field(default_factory=dict)
    execution_metadata: Dict[str, Any] = field(default_factory=dict)
    unavailable_reason: Optional[str] = None
    required_inputs_description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


class BaseSpecialist(ABC):
    """Abstract base class for all SatQuery AI analytical specialist modules."""

    def __init__(self, name: str, supported_tasks: List[TaskType], required_input_type: InputType):
        self.name = name
        self.supported_tasks = supported_tasks
        self.required_input_type = required_input_type

    @abstractmethod
    def supports_input(self, input_data: Any) -> Tuple[bool, Optional[str]]:
        """
        Check if input data matches this specialist's requirements.
        Returns: (is_compatible, error_message_if_not)
        """
        pass

    @abstractmethod
    def predict(self, input_data: Any) -> SpecialistResult:
        """Run inference and return structured SpecialistResult."""
        pass
