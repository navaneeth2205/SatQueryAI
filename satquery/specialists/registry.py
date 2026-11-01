"""
Specialist Model Registry for SatQuery AI.

Manages active specialist models, performs capability checks,
and provides explicit structured stubs for unsupported tasks.
"""

from typing import Dict, Optional, Any
from satquery.taxonomy import TaskType, TASK_REGISTRY
from satquery.specialists.base import BaseSpecialist, SpecialistResult, SpecialistStatus


class SpecialistRegistry:
    """
    Central registry for analytical specialist models.
    Enforces explicit capability checking and transparent unavailability reporting.
    """

    def __init__(self):
        self._specialists: Dict[TaskType, BaseSpecialist] = {}
        self._dispatch_cache: Dict[Any, SpecialistResult] = {}

    def clear_cache(self) -> None:
        """Clear the specialist execution cache."""
        self._dispatch_cache.clear()

    def register(self, specialist: BaseSpecialist) -> None:
        """Register a specialist for all tasks it explicitly supports."""
        for task in specialist.supported_tasks:
            self._specialists[task] = specialist
        self.clear_cache()

    def get_specialist(self, task: TaskType) -> Optional[BaseSpecialist]:
        """Retrieve the registered specialist for a task, if available."""
        return self._specialists.get(task)

    def dispatch(self, task: TaskType, input_data: Any = None) -> SpecialistResult:
        """
        Dispatch input to the registered specialist, or return an explicit UNAVAILABLE stub.
        """
        cache_key = (task.value, id(input_data))
        if cache_key in self._dispatch_cache:
            return self._dispatch_cache[cache_key]

        task_def = TASK_REGISTRY.get(task)
        display_name = task_def.display_name if task_def else task.value
        req_input_desc = (
            f"Requires {task_def.required_input.value.replace('_', ' ')} observation"
            if task_def else "Input requirements unspecified"
        )

        specialist = self.get_specialist(task)

        # 1. Handle Unsupported Tasks (Explicit Stubs)
        if specialist is None:
            planned_target = task_def.specialist_model_family if task_def else "specialist_model"
            res = SpecialistResult(
                status=SpecialistStatus.UNAVAILABLE,
                task=task.value,
                unavailable_reason=(
                    f"No active specialist model is currently registered for '{display_name}'. "
                    f"Planned specialist '{planned_target}' is scheduled for Milestone 3."
                ),
                required_inputs_description=req_input_desc,
                execution_metadata={
                    "registered_specialist": None,
                    "target_model_family": planned_target,
                    "is_stub": True
                }
            )
            self._dispatch_cache[cache_key] = res
            return res

        # 2. Check Input Compatibility
        is_compat, error_msg = specialist.supports_input(input_data)
        if not is_compat:
            res = SpecialistResult(
                status=SpecialistStatus.INCOMPATIBLE_INPUT,
                task=task.value,
                unavailable_reason=f"Incompatible input: {error_msg}",
                required_inputs_description=req_input_desc,
                execution_metadata={
                    "specialist_name": specialist.name,
                    "is_stub": False
                }
            )
            self._dispatch_cache[cache_key] = res
            return res

        # 3. Execute Validated Specialist
        res = specialist.predict(input_data)
        self._dispatch_cache[cache_key] = res
        return res
