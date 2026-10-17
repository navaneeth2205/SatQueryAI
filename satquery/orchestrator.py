"""
SatQuery AI Central Orchestration Engine.

Wires natural-language task routing, specialist model dispatching,
evidence formatting, and observable execution trace logging.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from satquery.taxonomy import TaskType, TASK_REGISTRY
from satquery.router import TaskRouter, ExecutionRoute
from satquery.specialists.base import SpecialistResult, SpecialistStatus
from satquery.specialists.registry import SpecialistRegistry
from satquery.specialists.land_cover import LandCoverClassifier
from satquery.evidence import format_evidence, EvidenceReport


@dataclass
class SatQueryResponse:
    """Complete user-facing response from SatQuery AI."""
    query: str
    status: str
    routed_task: str
    routed_task_display: str
    router_confidence: float
    requires_human_review: bool
    answer: str
    evidence: EvidenceReport
    execution_trace: List[str]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["evidence"] = self.evidence.to_dict()
        return data


class SatQueryEngine:
    """
    End-to-End Orchestrator for SatQuery AI.
    Coordinates router, specialist registry, and evidence formatter.
    """

    def __init__(self, router: TaskRouter, registry: SpecialistRegistry):
        self.router = router
        self.registry = registry

    @classmethod
    def create_default(
        cls,
        router_model_path: Optional[str | Path] = None,
        land_cover_artifacts_dir: Optional[str | Path] = None
    ) -> "SatQueryEngine":
        """Factory method building default engine with router and land cover specialist."""
        repo_root = Path(__file__).resolve().parent.parent

        if router_model_path is None:
            router_model_path = repo_root / "artifacts" / "router" / "router_model.joblib"

        if not Path(router_model_path).exists():
            raise FileNotFoundError(f"Router model artifact not found at {router_model_path}.")

        router = TaskRouter.load(router_model_path)

        registry = SpecialistRegistry()
        land_cover_specialist = LandCoverClassifier(artifacts_dir=land_cover_artifacts_dir)
        registry.register(land_cover_specialist)

        return cls(router=router, registry=registry)

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

        # 1. Run Task Router
        route: ExecutionRoute = self.router.predict(query)
        task_str = route.task
        trace.append(
            f"[Trace 2/5] Task Router classified intent as '{task_str}' with {route.confidence*100:.1f}% confidence."
        )

        if route.requires_review:
            trace.append(
                f"[Safety Trace] Routing confidence {route.confidence:.2f} is below automated threshold. Flagged for review."
            )

        # 2. Dispatch to Specialist via Registry
        try:
            task_enum = TaskType(task_str)
        except ValueError:
            task_enum = None

        if task_enum is None:
            specialist_result = SpecialistResult(
                status=SpecialistStatus.UNAVAILABLE,
                task=task_str,
                unavailable_reason=f"Unknown task enum '{task_str}'."
            )
        else:
            trace.append(f"[Trace 3/5] Dispatching to Specialist Registry for task '{task_enum.value}'...")
            specialist_result = self.registry.dispatch(task_enum, input_data)

        # 3. Log Specialist Status
        if specialist_result.status == SpecialistStatus.SUCCESS:
            trace.append(
                f"[Trace 4/5] Specialist '{specialist_result.execution_metadata.get('specialist_name')}' executed successfully."
            )
            overall_status = "SUCCESS"
        elif specialist_result.status == SpecialistStatus.INCOMPATIBLE_INPUT:
            trace.append(
                f"[Trace 4/5] Input Error: {specialist_result.unavailable_reason}."
            )
            overall_status = "INCOMPATIBLE_INPUT"
        else:
            trace.append(
                f"[Trace 4/5] Task Unavailable: {specialist_result.unavailable_reason}"
            )
            overall_status = "UNAVAILABLE_SPECIALIST"

        # 4. Format Evidence
        evidence_report = format_evidence(specialist_result)
        trace.append(f"[Trace 5/5] Formatted explainable answer with {len(evidence_report.evidence_points)} evidence points.")

        return SatQueryResponse(
            query=query,
            status=overall_status,
            routed_task=task_str,
            routed_task_display=route.display_name,
            router_confidence=route.confidence,
            requires_human_review=route.requires_review,
            answer=evidence_report.answer,
            evidence=evidence_report,
            execution_trace=trace,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
