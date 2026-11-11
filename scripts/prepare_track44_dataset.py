"""
Dataset Preparation Script for SatQuery AI Phase 4 Track 4.4:
Complex & Adversarial Linguistic Query Defense.

Governed strictly by SatQuery AI Phase 4 Frozen Protocol v1.12.1.
Generates:
- Exactly N = 125 Adversarial Queries across 5 Cohorts of N = 25:
  1. Cohort 1: Adversarial Lexical Perturbations (Ground Truth: REJECT_INSUFFICIENT_CONTEXT)
  2. Cohort 2: Conflicting Multi-Task Collisions (Ground Truth: CLARIFY_AMBIGUOUS)
  3. Cohort 3: Polysemous Domain Jargon (Ground Truth: CLARIFY_AMBIGUOUS)
  4. Cohort 4: Truncated / Incomplete Instructions (Ground Truth: REJECT_INSUFFICIENT_CONTEXT)
  5. Cohort 5: Negation Traps (Ground Truth: CLARIFY_AMBIGUOUS)
- Exactly N = 25 In-Domain Control Queries:
  - 8 Land Cover Classification queries
  - 8 Change Detection queries
  - 9 Other supported architectural intents (Change VQA, Single Image VQA, Scene Captioning, Visual Grounding, Optical-SAR Fusion)
Total: N = 150 benchmark queries stored in configs/track44_linguistic_defense_config.json
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "track44_linguistic_defense_config.json"

FROZEN_PROTOCOL_HASH = "7bcb8c5ab00cfebb5f7418237e9297f7eeaf06b0477d426d0ec6b30880e71bb5"


def build_track44_dataset() -> Dict[str, Any]:
    # Cohort 1: Adversarial Lexical Perturbations (N=25)
    # Ground Truth: REJECT_INSUFFICIENT_CONTEXT
    c1_queries = [
        "Classify the temporal modifications in this agricultural pasture tile.",
        "Detect SAR backscatter land-cover changes across this optical scene.",
        "Calculate bounding box pixel coordinates for temporal land cover deforestation.",
        "Segment the scene captioning change detection masks.",
        "Predict radar backscatter differences while classifying urban fabric.",
        "Run biophysical land use classification on satellite orbit ephemeris data.",
        "Determine whether pixel elevation contours represent forest or pasture land cover.",
        "Classify agricultural cropland using atmospheric carbon dioxide ppm concentration.",
        "Detect building alterations using ground penetrating radar audio acoustic logs.",
        "Estimate surface change detection from thermal infrared heat island sensor drift.",
        "Map CORINE biophysical categories using sub-orbital drone lidar telemetry packets.",
        "Identify arable crop fields from atmospheric aerosol optical depth lookup tables.",
        "Calculate deforestation rate using sea surface salinity buoys.",
        "Classify urban fabric using cellular network tower signal triangulation.",
        "Detect temporal surface change from gravity recovery and climate experiment GRACE mascons.",
        "Predict land cover vegetation using magnetosphere geomagnetic flux readings.",
        "Identify pasture land from Doppler radar weather reflectivity scans.",
        "Classify wetland biomes using seismic tremor magnitude recordings.",
        "Detect infrastructure modifications from AIS maritime ship beacon transponders.",
        "Determine cropland boundaries using stratospheric ozone depletion column density.",
        "Classify forest canopy cover from solar wind geomagnetic storm indices.",
        "Map water bodies using aircraft transponder flight path logs.",
        "Detect urban building growth from nightlight luminosity wattage meters.",
        "Classify industrial surfaces using ionospheric total electron content maps.",
        "Identify transitional woodland from soil moisture active passive SMAP radiometer counts."
    ]

    # Cohort 2: Conflicting Multi-Task Collisions (N=25)
    # Ground Truth: CLARIFY_AMBIGUOUS
    c2_queries = [
        "Classify land cover classes and detect modifications between 2020 and 2024 simultaneously.",
        "Determine both the CORINE land cover categories and temporal building changes over time.",
        "Perform optical-SAR fusion and bitemporal change detection on this observation.",
        "Execute land-cover classification and bounding box visual grounding on storage tanks.",
        "Provide a descriptive scene caption and detect all surface modifications across dates.",
        "Count airport aircraft and classify agricultural crop types in this image.",
        "Detect building change and count how many ships are docked in the harbor.",
        "Map forest and arable land cover while localizing coordinates for all runways.",
        "Generate a caption for the landscape and classify every pixel into CORINE classes.",
        "Perform visual grounding on industrial buildings and detect temporal modifications.",
        "Fuse radar and optical imagery to detect changed structures between observation dates.",
        "Answer how many buildings are damaged and classify surrounding biophysical land use.",
        "Detect deforestation over time and localize pixel coordinates of remaining tree stands.",
        "Describe this scene in detail and detect whether new buildings were constructed.",
        "Classify pastures and arable land while counting aircraft on the taxiway.",
        "Localize bounding box coordinates for vehicles and detect temporal road changes.",
        "Perform SAR polarization fusion while identifying crop types and urban sprawl.",
        "Detect surface alterations between dates and provide a concise caption of the area.",
        "Classify water bodies and wetlands while answering how many bridges are visible.",
        "Identify urban fabric and calculate bounding box coordinates for fuel tanks.",
        "Detect newly built structures over time and classify the underlying soil biome.",
        "Describe the terrain landscape and execute optical-SAR backscatter fusion.",
        "Classify cropland and detect modification between pre-event and post-event passes.",
        "Localize all aircraft with bounding boxes and detect ground changes since 2021.",
        "Determine biophysical land cover categories and count storage containers in the yard."
    ]

    # Cohort 3: Polysemous Domain Jargon (N=25)
    # Ground Truth: CLARIFY_AMBIGUOUS
    c3_queries = [
        "Spectral reflectance signatures across the biophysical canopy.",
        "Top-of-atmosphere radiance and bottom-of-atmosphere surface reflectance.",
        "Radiometric calibration profiles and Sentinel-2 Level-2A metadata.",
        "Geospatial coordinate reference system EPSG:32632 projection parameters.",
        "High-resolution orthorectified multispectral imagery.",
        "Atmospheric transmission windows and Rayleigh scattering coefficients.",
        "Sentinel-1 Synthetic Aperture Radar interferometric coherence phase.",
        "Ground sample distance and subpixel spatial resolution characteristics.",
        "Sun glint specular reflection and bidirectional reflectance distribution function.",
        "Digital elevation model slope, aspect, and topographic terrain contours.",
        "Normalized Difference Vegetation Index red and near-infrared spectral bands.",
        "Cloud shadow radiometric overcorrection and dark pixel atmospheric correction.",
        "Shortwave infrared SWIR band atmospheric aerosol absorption profiles.",
        "Copernicus Space Component Sentinel-2 MSI instrument spectral response functions.",
        "Dual polarization VV and VH backscatter scattering matrix components.",
        "Orthorectification rational polynomial coefficients and sensor telemetry.",
        "Radiometric quantization levels and 12-bit digital number scaling offsets.",
        "Spatial modulation transfer function and point spread function calibration.",
        "Temporal revisit frequency and sun-synchronous orbital ground track.",
        "Geometric distortion affine transformation matrix coefficients.",
        "Albedo reflectance across visible blue green and red channels.",
        "Panchromatic sharpening spectral band resampling algorithms.",
        "Thermal infrared radiative heat transfer equations.",
        "Multispectral swath width and nadir viewing angle geometry.",
        "Atmospheric water vapor column absorption and cirrus cloud screening."
    ]

    # Cohort 4: Truncated / Incomplete Instructions (N=25)
    # Ground Truth: REJECT_INSUFFICIENT_CONTEXT
    c4_queries = [
        "change",
        "forest",
        "satellite",
        "detect",
        "analyze",
        "pasture",
        "run model",
        "2024",
        "Sentinel-2",
        "imagery",
        "classify",
        "tile",
        "patch",
        "view",
        "process",
        "urban",
        "crop",
        "data",
        "run",
        "what is",
        "compare",
        "image",
        "difference",
        "map",
        "help"
    ]

    # Cohort 5: Negation Traps (N=25)
    # Ground Truth: CLARIFY_AMBIGUOUS
    c5_queries = [
        "Do not detect changes, only inspect the static satellite image.",
        "Do not classify land cover, ignore all vegetation and forest types.",
        "Never perform change detection between these two acquisition dates.",
        "Avoid land cover classification and do not categorize biophysical surfaces.",
        "Do not localize bounding boxes or count objects in this image.",
        "Do not detect modifications over time, just look at the single picture.",
        "Skip change detection completely and refrain from finding alterations.",
        "Never classify crop types or arable agricultural land in this tile.",
        "Do not run land-cover classification on this observation.",
        "Refrain from detecting temporal changes across the years.",
        "Avoid classifying urban fabric and pastures in this scene.",
        "Do not detect new building constructions between the dates.",
        "Do not produce a land cover map for this satellite patch.",
        "Never identify deforestation or vegetation change in this area.",
        "Do not perform optical-SAR fusion on these radar bands.",
        "Avoid detecting any modifications between pre-event and post-event imagery.",
        "Refrain from classifying the biome or soil category here.",
        "Do not highlight change detection areas between 2020 and 2024.",
        "Never classify this scene as forest or cropland.",
        "Do not look for alterations between the earlier and later images.",
        "Avoid identifying biophysical land cover classes in this region.",
        "Do not count aircraft on the runway and avoid VQA.",
        "Refrain from detecting ground sprawl or temporal construction.",
        "Do not classify whether this area is pasture or arable land.",
        "Never run change detection on this bitemporal pair."
    ]

    # In-Domain Control Cohort (N=25)
    # Ground Truth: ROUTED_TO_INTENT (Specific specialist target)
    control_queries = [
        # 8 Land Cover Classification queries
        {"query": "What type of land cover is visible in this satellite patch?", "expected_task": "LAND_COVER_CLASSIFICATION"},
        {"query": "Classify the surface terrain and vegetation cover.", "expected_task": "LAND_COVER_CLASSIFICATION"},
        {"query": "Identify whether this region contains arable land or forest.", "expected_task": "LAND_COVER_CLASSIFICATION"},
        {"query": "Determine the distribution of pastures and crop fields.", "expected_task": "LAND_COVER_CLASSIFICATION"},
        {"query": "Classify forest, arable land, and pastures in this scene.", "expected_task": "LAND_COVER_CLASSIFICATION"},
        {"query": "What land-cover classes are present in this agricultural tile?", "expected_task": "LAND_COVER_CLASSIFICATION"},
        {"query": "Classify vegetation types and biophysical land cover.", "expected_task": "LAND_COVER_CLASSIFICATION"},
        {"query": "Map the distribution of arable land and coniferous forest.", "expected_task": "LAND_COVER_CLASSIFICATION"},
        # 8 Change Detection queries
        {"query": "What changed in this area between 2024 and 2026?", "expected_task": "CHANGE_DETECTION"},
        {"query": "Detect land-cover modifications between these two acquisition dates.", "expected_task": "CHANGE_DETECTION"},
        {"query": "Has urban expansion occurred between time T1 and T2?", "expected_task": "CHANGE_DETECTION"},
        {"query": "Identify ground modifications between pre-event and post-event passes.", "expected_task": "CHANGE_DETECTION"},
        {"query": "What changed between the 2021 and 2024 acquisitions?", "expected_task": "CHANGE_DETECTION"},
        {"query": "Detect building construction and surface modifications over time.", "expected_task": "CHANGE_DETECTION"},
        {"query": "Identify newly built structures between these two observation dates.", "expected_task": "CHANGE_DETECTION"},
        {"query": "Detect deforestation and surface alterations across this bitemporal pair.", "expected_task": "CHANGE_DETECTION"},
        # 9 Other Supported Architectural Intents
        {"query": "Did the lake surface area shrink between the two dates?", "expected_task": "CHANGE_VQA"},
        {"query": "How much has urban construction grown compared to the previous year?", "expected_task": "CHANGE_VQA"},
        {"query": "Why does the vegetation index appear lower in the second image?", "expected_task": "CHANGE_VQA"},
        {"query": "How many aircraft are parked on the airfield runway?", "expected_task": "SINGLE_IMAGE_VQA"},
        {"query": "Is there a river or reservoir present in the southern portion?", "expected_task": "SINGLE_IMAGE_VQA"},
        {"query": "Generate a descriptive summary of this satellite scene.", "expected_task": "SCENE_CAPTIONING"},
        {"query": "Describe the geographical landscape shown in this image.", "expected_task": "SCENE_CAPTIONING"},
        {"query": "Locate the reservoir and highlight its boundary with a mask.", "expected_task": "VISUAL_GROUNDING"},
        {"query": "Compare radar backscatter and optical reflectance for this scene.", "expected_task": "OPTICAL_SAR_FUSION"},
    ]

    # Package into structured benchmark format
    cohort_specs = [
        {
            "cohort_id": "cohort_1_lexical_perturbations",
            "name": "Cohort 1: Adversarial Lexical Perturbations",
            "threat_type": "Conflicting task keywords and out-of-domain vocabulary spoofing TF-IDF balances",
            "ground_truth_action": "REJECT_INSUFFICIENT_CONTEXT",
            "queries": [{"query_id": f"C1_{i+1:02d}", "query": q, "ground_truth": "REJECT_INSUFFICIENT_CONTEXT"} for i, q in enumerate(c1_queries)]
        },
        {
            "cohort_id": "cohort_2_multi_task_collisions",
            "name": "Cohort 2: Conflicting Multi-Task Collisions",
            "threat_type": "Blended mutually exclusive task verbs demanding simultaneous execution",
            "ground_truth_action": "CLARIFY_AMBIGUOUS",
            "queries": [{"query_id": f"C2_{i+1:02d}", "query": q, "ground_truth": "CLARIFY_AMBIGUOUS"} for i, q in enumerate(c2_queries)]
        },
        {
            "cohort_id": "cohort_3_polysemous_jargon",
            "name": "Cohort 3: Polysemous Domain Jargon",
            "threat_type": "Geospatial and radiometric terminology devoid of actionable operational directives",
            "ground_truth_action": "CLARIFY_AMBIGUOUS",
            "queries": [{"query_id": f"C3_{i+1:02d}", "query": q, "ground_truth": "CLARIFY_AMBIGUOUS"} for i, q in enumerate(c3_queries)]
        },
        {
            "cohort_id": "cohort_4_truncated_instructions",
            "name": "Cohort 4: Truncated / Incomplete Instructions",
            "threat_type": "Syntactically broken strings, fragments, abbreviations, or single-token queries",
            "ground_truth_action": "REJECT_INSUFFICIENT_CONTEXT",
            "queries": [{"query_id": f"C4_{i+1:02d}", "query": q, "ground_truth": "REJECT_INSUFFICIENT_CONTEXT"} for i, q in enumerate(c4_queries)]
        },
        {
            "cohort_id": "cohort_5_negation_traps",
            "name": "Cohort 5: Negation Traps",
            "threat_type": "Semantic negation clauses unresolvable by linear bag-of-words representations",
            "ground_truth_action": "CLARIFY_AMBIGUOUS",
            "queries": [{"query_id": f"C5_{i+1:02d}", "query": q, "ground_truth": "CLARIFY_AMBIGUOUS"} for i, q in enumerate(c5_queries)]
        },
    ]

    control_spec = {
        "cohort_id": "cohort_control_in_domain",
        "name": "In-Domain Control Cohort",
        "threat_type": "Legitimate unambiguous in-domain satellite analysis queries",
        "ground_truth_action": "ROUTED_TO_INTENT",
        "queries": [{"query_id": f"CTRL_{i+1:02d}", "query": item["query"], "expected_task": item["expected_task"], "ground_truth": "ROUTED_TO_INTENT"} for i, item in enumerate(control_queries)]
    }

    dataset = {
        "metadata": {
            "track": "Track 4.4: Complex & Adversarial Linguistic Query Defense",
            "protocol_reference": "SATQUERY-P4-PROTOCOL-V1.12.1-FROZEN.md",
            "protocol_hash": FROZEN_PROTOCOL_HASH,
            "threat_model": "TF-IDF + Calibrated Logistic Regression feature space exploitation",
            "calibration_temperature": 0.0991,
            "tau_intent": 0.65,
            "tau_margin": 0.15,
            "tau_sel": 1.0,
            "total_queries": 150,
            "n_adversarial_queries": 125,
            "n_control_queries": 25,
            "n_adversarial_cohorts": 5,
            "queries_per_adversarial_cohort": 25,
            "certification_tiers": {
                "Tier_1": {
                    "description": "Certified Robust Pass",
                    "pooled_fdr_max_pct": 10.0,
                    "pooled_max_false_dispatches": 12,
                    "cohort_max_false_dispatches": 5,
                    "min_control_preservation": 24
                },
                "Tier_2": {
                    "description": "Operational with Subsystem Reservation",
                    "pooled_fdr_max_pct": 12.0,
                    "pooled_max_false_dispatches": 15,
                    "cohort_max_false_dispatches": 6,
                    "min_control_preservation": 23
                },
                "Tier_3": {
                    "description": "Failed Certification",
                    "condition": "pooled_false_dispatches >= 16 OR max_kc > 6 OR control_preservation < 23"
                }
            }
        },
        "adversarial_cohorts": cohort_specs,
        "control_cohort": control_spec
    }

    return dataset


def main():
    print("Generating Track 4.4 Fixed Adversarial Linguistic Benchmark Dataset...")
    dataset = build_track44_dataset()
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    with open(CONFIG_PATH, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()

    print(f"Generated {dataset['metadata']['total_queries']} benchmark queries.")
    print(f"Saved to: {CONFIG_PATH}")
    print(f"SHA-256 Digest: {digest}")


if __name__ == "__main__":
    main()
