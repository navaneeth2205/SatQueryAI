"""
End-to-End Pipeline Orchestration Engine for SatQuery AI.

Coordinates:
- Natural language query ingestion
- Input observation metadata extraction (modality, validity, sensor verification)
- Compatibility-aware task routing (separating linguistic confidence from input compatibility)
- Safety policy arbitration ('provisional' vs 'clarify')
- Specialist registry dispatch (Land Cover, Change Detection, and honest unavailable stubs)
- Explainable evidence synthesis with observable execution trace
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from satquery.taxonomy import TaskType, InputType, TASK_REGISTRY, get_task_definition
from satquery.router import TaskRouter, ExecutionRoute
from satquery.specialists.base import SpecialistResult, SpecialistStatus
from satquery.specialists.registry import SpecialistRegistry
from satquery.specialists.land_cover import LandCoverClassifier
from satquery.specialists.change_detection import TemporalDifferenceBaseline
from satquery.evidence import format_evidence, EvidenceReport
from satquery.utils.metadata import (
    InputObservationMetadata,
    extract_observation_metadata,
    TASK_COMPATIBILITY_MAP
)


@dataclass
class SatQueryResponse:
    """Complete user-facing response from SatQuery AI."""
    query: str
    status: str
    execution_policy: str
    routed_task: str
    routed_task_display: str
    router_confidence: float
    requires_human_review: bool
    answer: str
    evidence: EvidenceReport
    execution_trace: List[str]
    timestamp: str
    # Scientific refinement fields:
    text_task: str = ""
    text_confidence: float = 0.0
    selected_task: str = ""
    input_compatibility: bool = True
    routing_reason: str = ""
    attached_modality: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["evidence"] = self.evidence.to_dict()
        return data


class SatQueryEngine:
    """
    End-to-End Orchestrator for SatQuery AI.
    Coordinates router, specialist registry, and evidence formatter.
    Supports configurable routing policies: 'provisional' vs 'clarify'.
    """

    def __init__(
        self,
        router: TaskRouter,
        registry: SpecialistRegistry,
        routing_policy: str = "provisional"
    ):
        self.router = router
        self.registry = registry
        if routing_policy not in ("provisional", "clarify"):
            raise ValueError(f"Unknown routing policy '{routing_policy}'. Expected 'provisional' or 'clarify'.")
        self.routing_policy = routing_policy

    @classmethod
    def create_default(
        cls,
        router_path: Optional[str | Path] = None,
        routing_policy: str = "provisional"
    ) -> "SatQueryEngine":
        """
        Factory method instantiating the default production SatQueryEngine.
        Loads trained router and registers all implemented specialist models.
        """
        repo_root = Path(__file__).resolve().parent.parent

        if router_path is None:
            router_path = repo_root / "artifacts" / "router" / "router_model.joblib"

        router = TaskRouter.load(router_path)

        registry = SpecialistRegistry()
        # Register Land Cover Classifier (Milestone 2)
        lc_specialist = LandCoverClassifier()
        registry.register(lc_specialist)

        # Register Multitemporal Change Detection Specialist (Milestone 3)
        cd_specialist = TemporalDifferenceBaseline()
        registry.register(cd_specialist)

        return cls(router=router, registry=registry, routing_policy=routing_policy)

    def process_query(
        self,
        query: str,
        input_data: Any = None
    ) -> SatQueryResponse:
        """
        Process user query and input imagery end-to-end.
        """
        trace: List[str] = [
            f"[Trace 1/5] Ingested query: \"{query}\"",
        ]

        # 1. Extract observation metadata from input_data
        obs_meta = extract_observation_metadata(input_data)

        # 2. Run Compatibility-Aware Task Router
        route: ExecutionRoute = self.router.predict(query, image_metadata=obs_meta)
        task_str = route.task
        trace.append(
            f"[Trace 2/5] Task Router classified intent as '{task_str}' with {route.confidence*100:.1f}% confidence."
        )
        if not route.input_compatibility:
            trace.append(f"[Disambiguation Trace] {route.routing_reason}")

        # 3. Evaluate Routing Policy (tied strictly to linguistic confidence)
        if route.requires_review:
            if self.routing_policy == "clarify":
                execution_policy = "CLARIFICATION_REQUIRED"
                trace.append(
                    f"[Safety Trace] Routing confidence {route.confidence*100:.1f}% is below threshold ({self.router.confidence_threshold*100:.1f}%). "
                    f"Policy: CLARIFICATION_REQUIRED — Automated execution halted. User clarification requested."
                )
                clarification_answer = (
                    f"[CLARIFICATION REQUIRED: Task routing confidence is {route.confidence*100:.1f}%, which is below the review threshold of {self.router.confidence_threshold*100:.1f}%. "
                    f"The router tentatively identified '{route.display_name}', but please clarify or confirm your analytical goal before automated specialist execution.]"
                )
                empty_report = EvidenceReport(
                    answer=clarification_answer,
                    status="NEEDS_CLARIFICATION",
                    detected_classes=[],
                    class_probabilities={},
                    thresholds_applied={},
                    evidence_points=[
                        f"[Status] NEEDS_CLARIFICATION",
                        f"[Confidence] {route.confidence*100:.1f}% (threshold: {self.router.confidence_threshold*100:.1f}%)",
                        f"[Tentative Task] {route.display_name} ({task_str})"
                    ],
                    sensor_feature_summary={"route": route.to_dict()}
                )
                return SatQueryResponse(
                    query=query,
                    status="NEEDS_CLARIFICATION",
                    execution_policy=execution_policy,
                    routed_task=task_str,
                    routed_task_display=route.display_name,
                    router_confidence=route.confidence,
                    requires_human_review=True,
                    answer=clarification_answer,
                    evidence=empty_report,
                    execution_trace=trace,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    text_task=route.text_task,
                    text_confidence=route.text_confidence,
                    selected_task=route.selected_task,
                    input_compatibility=route.input_compatibility,
                    routing_reason=route.routing_reason,
                    attached_modality=route.attached_modality
                )
            else:
                execution_policy = "PROVISIONAL_EXECUTION"
                trace.append(
                    f"[Safety Trace] Routing confidence {route.confidence*100:.1f}% is below threshold ({self.router.confidence_threshold*100:.1f}%). "
                    f"Policy: PROVISIONAL_EXECUTION — specialist executes provisionally, but routing intent is flagged for human review. "
                    f"(Note: Task routing confidence measures user intent clarity, distinct from specialist prediction probabilities.)"
                )
        else:
            execution_policy = "STANDARD_EXECUTION"

        # 4. Dispatch to Specialist via Registry
        try:
            task_enum = TaskType(task_str)
        except ValueError:
            task_enum = None

        if task_enum is None:
            specialist_result = SpecialistResult(
                status=SpecialistStatus.UNAVAILABLE,
                task=task_str,
                unavailable_reason=f"Task '{task_str}' is not recognized in the active taxonomy."
            )
        else:
            trace.append(f"[Trace 3/5] Dispatching to Specialist Registry for task '{task_str}'...")
            specialist_result = self.registry.dispatch(task_enum, input_data)

        # 5. Format Evidence and Handle Specialist Status
        if specialist_result.status == SpecialistStatus.UNAVAILABLE:
            status = "UNAVAILABLE_SPECIALIST"
            trace.append(f"[Trace 4/5] Task Unavailable: {specialist_result.unavailable_reason}")
            evidence = format_evidence(specialist_result)

        elif specialist_result.status == SpecialistStatus.INCOMPATIBLE_INPUT:
            status = "INCOMPATIBLE_INPUT"
            trace.append(f"[Trace 4/5] Input Error: {specialist_result.unavailable_reason}.")
            evidence = format_evidence(specialist_result)

        elif specialist_result.status == SpecialistStatus.SUCCESS:
            status = "PROVISIONAL_SUCCESS" if execution_policy == "PROVISIONAL_EXECUTION" else "SUCCESS"
            specialist_name = specialist_result.execution_metadata.get("specialist_name", "RegisteredSpecialist")
            trace.append(f"[Trace 4/5] Specialist '{specialist_name}' executed successfully.")
            evidence = format_evidence(specialist_result)

            # Prefix answer with provisional execution disclaimer if policy was provisional
            if execution_policy == "PROVISIONAL_EXECUTION":
                evidence.answer = (
                    f"[PROVISIONAL ROUTE: Routing intent confidence is {route.confidence*100:.1f}%, below the review threshold of {self.router.confidence_threshold*100:.1f}%. "
                    f"Specialist executed provisionally. Scores shown are model voting probabilities, not routing confidence.] "
                    + evidence.answer
                )
        else:
            status = "FAILED"
            trace.append(f"[Trace 4/5] Unknown specialist status: {specialist_result.status}.")
            evidence = format_evidence(specialist_result)

        trace.append(f"[Trace 5/5] Formatted explainable answer with {len(evidence.evidence_points)} evidence points.")

        return SatQueryResponse(
            query=query,
            status=status,
            execution_policy=execution_policy,
            routed_task=task_str,
            routed_task_display=route.display_name,
            router_confidence=route.confidence,
            requires_human_review=route.requires_review,
            answer=evidence.answer,
            evidence=evidence,
            execution_trace=trace,
            timestamp=datetime.now(timezone.utc).isoformat(),
            text_task=route.text_task,
            text_confidence=route.text_confidence,
            selected_task=route.selected_task,
            input_compatibility=route.input_compatibility,
            routing_reason=route.routing_reason,
            attached_modality=route.attached_modality
        )
