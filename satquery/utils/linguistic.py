"""
Linguistic Defense Gate for SatQuery AI (Phase 4 Track 4.4).

Governed strictly by SatQuery AI Phase 4 Frozen Protocol v1.12.1.
Protects specialist models from adversarial linguistic exploitation targeting
linear TF-IDF + Logistic Regression feature spaces:
1. Multi-task lexical collisions (coordinating mutually exclusive task verbs/keywords)
2. Negation traps (semantic negation clauses unresolvable by bag-of-words)
3. Polysemous domain jargon devoid of actionable operational directives
4. Syntactically broken, truncated, or incomplete instructions
5. Out-of-domain telemetry and non-imaging physical sensors
"""

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class LinguisticDefenseResult:
    """Diagnostic outcome of the Linguistic Defense Gate."""
    query: str
    is_intercepted: bool
    decision: str                      # "PASS", "CLARIFY_AMBIGUOUS", "REJECT_INSUFFICIENT_CONTEXT"
    failure_code: Optional[str]        # None, "ERR_MULTI_TASK_COLLISION", "ERR_NEGATION_TRAP", etc.
    threat_description: Optional[str]
    matched_tasks: List[str]
    detected_keywords: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "is_intercepted": self.is_intercepted,
            "decision": self.decision,
            "failure_code": self.failure_code,
            "threat_description": self.threat_description,
            "matched_tasks": self.matched_tasks,
            "detected_keywords": self.detected_keywords
        }


# Out-of-domain sensors and non-imaging physics
OOD_TELEMETRY_TERMS: List[str] = [
    "seismic", "geomagnetic", "ephemeris", "transponder", "telemetry",
    "cellular", "acoustic", "salinity", "mascons", "solar wind",
    "ionospheric", "buoys", "wattage meters", "geomagnetic flux",
    "radio reflectivity", "gravity recovery", "smap", "grace", "doppler",
    "stratospheric", "ozone depletion"
]

# Negation clauses directly modifying analytical intent
NEGATION_REGEX = re.compile(
    r"\b(do not|don't|never|avoid|refrain|skip|without|exclude|neither|nor|refrain from)\b",
    re.IGNORECASE
)

# Actionable remote sensing task verbs
DIRECTIVE_VERBS: List[str] = [
    "classify", "detect", "identify", "determine", "map", "count",
    "describe", "summarize", "localize", "locate", "find", "segment",
    "calculate", "estimate", "compare", "generate", "predict", "answer",
    "what", "how", "is", "did", "why", "were", "has"
]

# Primary specialist action verbs used in multi-task coordination checks
TASK_ACTION_VERBS: List[str] = [
    "classify", "detect", "count", "describe", "localize", "fuse",
    "caption", "grounding", "segment", "counting"
]

# Task distinctive keyword taxonomy matching satquery/router.py
TASK_KEYWORDS: Dict[str, List[str]] = {
    "LAND_COVER_CLASSIFICATION": [
        "land cover", "land-cover", "landcover", "corine", "biome",
        "forest", "arable", "pasture", "pastures", "crop", "cropland",
        "urban fabric", "biophysical", "vegetation", "coniferous forest",
        "arable land", "water bodies", "wetlands"
    ],
    "CHANGE_DETECTION": [
        "temporal modifications", "surface alterations", "between dates",
        "between 2020 and 2024", "over time", "deforestation over time",
        "newly built structures", "bitemporal", "across years", "since 2021",
        "pre-event and post-event", "temporal changes", "building changes",
        "temporal road changes", "ground changes", "modifications over time"
    ],
    "CHANGE_VQA": [
        "shrunk", "damaged", "destroyed", "expanded between", "disappeared",
        "how much has urban construction grown"
    ],
    "OPTICAL_SAR_FUSION": [
        "optical-sar", "optical and sar", "radar and optical",
        "sar backscatter", "polarization fusion", "sentinel-1 and sentinel-2",
        "radar backscatter and optical"
    ],
    "SINGLE_IMAGE_VQA": [
        "how many distinct water bodies", "is there a river", "airport runway",
        "aircraft on the runway", "storage tank visible", "how many aircraft",
        "count airport aircraft", "count how many", "count storage containers",
        "counting aircraft", "how many bridges"
    ],
    "SCENE_CAPTIONING": [
        "descriptive summary", "geographical landscape", "detailed textual caption",
        "provide a descriptive scene caption", "generate a caption for the landscape",
        "describe this scene in detail", "provide a concise caption",
        "describe the terrain landscape", "caption for the landscape"
    ],
    "VISUAL_GROUNDING": [
        "bounding box", "bbox", "pixel coordinates", "visual grounding",
        "localize coordinates", "localize all aircraft", "localizing coordinates",
        "bounding boxes for all storage tanks"
    ]
}


def evaluate_linguistic_defense_gate(query: str) -> LinguisticDefenseResult:
    """
    Evaluates a user query against the Linguistic Defense Gate.
    
    Returns:
        LinguisticDefenseResult with decision, failure code, and diagnostic metadata.
    """
    q_clean = query.strip()
    words = re.findall(r"\b\w+\b", q_clean.lower())

    # 1. Dimension 4: Instruction Completeness & Truncation Screening
    if len(words) < 3:
        return LinguisticDefenseResult(
            query=query,
            is_intercepted=True,
            decision="REJECT_INSUFFICIENT_CONTEXT",
            failure_code="ERR_INSUFFICIENT_CONTEXT",
            threat_description="Query is syntactically broken, truncated, or a single token lacking sufficient context.",
            matched_tasks=[],
            detected_keywords=words
        )

    # 2. Dimension 1: Out-of-Domain Non-Imaging Telemetry Spoofing
    detected_telemetry = [term for term in OOD_TELEMETRY_TERMS if term in q_clean.lower()]
    if detected_telemetry:
        return LinguisticDefenseResult(
            query=query,
            is_intercepted=True,
            decision="REJECT_INSUFFICIENT_CONTEXT",
            failure_code="ERR_OUT_OF_DOMAIN_TELEMETRY",
            threat_description="Query references non-imaging physical telemetry incompatible with satellite optical/SAR perception.",
            matched_tasks=[],
            detected_keywords=detected_telemetry
        )

    # 3. Dimension 5: Semantic Negation Trap Screening
    if NEGATION_REGEX.search(q_clean):
        return LinguisticDefenseResult(
            query=query,
            is_intercepted=True,
            decision="CLARIFY_AMBIGUOUS",
            failure_code="ERR_NEGATION_TRAP",
            threat_description="Semantic negation clause unresolvable by linear bag-of-words text representations.",
            matched_tasks=[],
            detected_keywords=re.findall(NEGATION_REGEX, q_clean)
        )

    # 4. Dimension 2: Multi-Task Coordination & Lexical Collision Screening
    found_task_verbs = [
        v for v in TASK_ACTION_VERBS
        if re.search(r"\b" + re.escape(v), q_clean, re.IGNORECASE)
    ]
    has_coordination = (
        len(found_task_verbs) >= 2
        and any(w in q_clean.lower() for w in ["while", " and ", "simultaneously", "both", "alongside"])
    )

    matched_tasks = []
    matched_kws = []
    for t_name, kws in TASK_KEYWORDS.items():
        found = [kw for kw in kws if re.search(r"\b" + re.escape(kw) + r"\b", q_clean, re.IGNORECASE)]
        if found:
            matched_tasks.append(t_name)
            matched_kws.extend(found)

    # If coordination between distinct task verbs, or >= 2 tasks have keyword matches
    if has_coordination or len(matched_tasks) >= 2:
        return LinguisticDefenseResult(
            query=query,
            is_intercepted=True,
            decision="CLARIFY_AMBIGUOUS",
            failure_code="ERR_MULTI_TASK_COLLISION",
            threat_description="Query combines vocabulary from multiple mutually exclusive specialist tasks.",
            matched_tasks=matched_tasks,
            detected_keywords=matched_kws
        )

    # 5. Dimension 3: Polysemous Domain Jargon Devoid of Operational Directives
    has_directive_verb = any(
        re.search(r"\b" + re.escape(v) + r"\b", q_clean, re.IGNORECASE)
        for v in DIRECTIVE_VERBS
    )
    if not has_directive_verb:
        return LinguisticDefenseResult(
            query=query,
            is_intercepted=True,
            decision="CLARIFY_AMBIGUOUS",
            failure_code="ERR_DOMAIN_JARGON_NO_VERB",
            threat_description="Geospatial domain terminology devoid of actionable operational directive or question verb.",
            matched_tasks=[],
            detected_keywords=words[:5]
        )

    # Legitimate un-intercepted query passes to calibrated Task Router
    return LinguisticDefenseResult(
        query=query,
        is_intercepted=False,
        decision="PASS",
        failure_code=None,
        threat_description=None,
        matched_tasks=matched_tasks,
        detected_keywords=matched_kws
    )
