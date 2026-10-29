"""
Expanded Task Router Robustness & Out-Of-Distribution (OOD) Evaluation.

Evaluates:
1. In-Distribution Paraphrase Robustness (N=42 diverse phrased queries across 7 tasks)
2. Ambiguous & Multi-Intent Queries (N=14)
3. Out-Of-Distribution & Non-EO Prompts (N=14)
4. Safety Review Filter Precision & Recall (Evaluating the 0.65 threshold as an anomaly/uncertainty detector)

Outputs:
- artifacts/benchmarks/router_expanded_robustness_summary.json
- artifacts/benchmarks/router_expanded_test_cases.csv
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from satquery.router import TaskRouter
from satquery.taxonomy import TaskType


def run_expanded_evaluation():
    print("=" * 75)
    print("      SATQUERY AI — EXPANDED ROUTER ROBUSTNESS & OOD BENCHMARK")
    print("=" * 75)

    router_path = REPO_ROOT / "artifacts" / "router" / "router_model.joblib"
    router = TaskRouter.load(router_path)

    # 1. In-Distribution Paraphrases (Diverse phrasing for all 7 tasks)
    in_distribution_cases = [
        # LAND_COVER_CLASSIFICATION
        ("Catalog every biophysical surface category visible in this tile.", "LAND_COVER_CLASSIFICATION"),
        ("What proportion of this Sentinel-2 footprint is occupied by coniferous forest?", "LAND_COVER_CLASSIFICATION"),
        ("Determine whether this rural area contains pastures or arable agricultural fields.", "LAND_COVER_CLASSIFICATION"),
        ("Segment and report urban fabric, industrial complexes, and inland water bodies.", "LAND_COVER_CLASSIFICATION"),
        ("Identify artificial surfaces and transportation infrastructure across this scene.", "LAND_COVER_CLASSIFICATION"),
        ("Classify all CORINE land-cover classes present in this multispectral observation.", "LAND_COVER_CLASSIFICATION"),

        # CHANGE_DETECTION
        ("Map ground modifications and construction activity between the two timestamps.", "CHANGE_DETECTION"),
        ("Did any forest clearing or deforestation occur between 2017 and 2018?", "CHANGE_DETECTION"),
        ("Compute Change Vector Analysis to identify altered surface patches.", "CHANGE_DETECTION"),
        ("Quantify the surface area of newly built structures across this temporal interval.", "CHANGE_DETECTION"),
        ("Detect significant spectral differences between the pre-event and post-event images.", "CHANGE_DETECTION"),
        ("Highlight all locations where vegetation cover was replaced by urban sprawl.", "CHANGE_DETECTION"),

        # OPTICAL_SAR_FUSION
        ("Integrate Sentinel-1 C-band SAR backscatter with Sentinel-2 optical bands.", "OPTICAL_SAR_FUSION"),
        ("Produce a fused multimodal radar-optical composite image.", "OPTICAL_SAR_FUSION"),
        ("Fuse VH/VV polarizations with optical RGB channels to improve feature contrast.", "OPTICAL_SAR_FUSION"),
        ("Combine microwave penetration data with multispectral reflectance.", "OPTICAL_SAR_FUSION"),

        # CHANGE_VQA
        ("Has the central water reservoir shrunk in size compared to the previous acquisition?", "CHANGE_VQA"),
        ("Were the bridge structures damaged following the flood event?", "CHANGE_VQA"),
        ("How many new buildings were erected in this district over the two-year period?", "CHANGE_VQA"),
        ("Is there evidence of agricultural expansion into the adjacent wetland?", "CHANGE_VQA"),

        # SINGLE_IMAGE_VQA
        ("How many cargo ships are docked at the deep-water port terminal?", "SINGLE_IMAGE_VQA"),
        ("What is the primary color of the commercial warehouse roofs in this industrial park?", "SINGLE_IMAGE_VQA"),
        ("Is there an airport runway clearly visible within this satellite image?", "SINGLE_IMAGE_VQA"),
        ("Are the agricultural fields organized in a circular center-pivot irrigation pattern?", "SINGLE_IMAGE_VQA"),

        # SCENE_CAPTIONING
        ("Provide a comprehensive descriptive caption of this coastal harbor scene.", "SCENE_CAPTIONING"),
        ("Summarize the landscape characteristics and human settlements in this tile.", "SCENE_CAPTIONING"),
        ("Generate an analytical overview paragraph describing the topography and land use.", "SCENE_CAPTIONING"),
        ("Describe in full sentences the terrain, waterways, and vegetation cover.", "SCENE_CAPTIONING"),

        # VISUAL_GROUNDING
        ("Locate the offshore oil drilling platform and return its pixel bounding box.", "VISUAL_GROUNDING"),
        ("Highlight the solar panel farm in the southern sector with a segmentation mask.", "VISUAL_GROUNDING"),
        ("Detect and outline the football stadium located near the urban center.", "VISUAL_GROUNDING"),
        ("Generate spatial coordinates for the major river intersection in this granule.", "VISUAL_GROUNDING")
    ]

    # 2. Ambiguous & Multi-Intent Queries
    ambiguous_cases = [
        ("Analyze this scene and tell me what changed or what trees exist.", "Multi-Intent (Change + Land Cover)"),
        ("Show me the satellite picture.", "Terse / Missing Action"),
        ("Check the status of this region.", "Vague / Undefined Analytic Task"),
        ("Is there any activity or just normal landscape?", "Multi-Intent (Change vs Static)"),
        ("Assess this footprint for environmental and structural patterns.", "Broad / Non-Specific"),
        ("Compare these images and explain what kind of land this is.", "Multi-Intent (Compare vs Classify)"),
        ("Examine radar and optical variations and report anything noticeable.", "Broad Fusion / General"),
        ("Look at the temporal pair and describe the scene in words.", "Multi-Intent (Temporal + Captioning)"),
        ("Find the buildings and tell me if they changed over time.", "Multi-Intent (Grounding + Change)"),
        ("Is this area developing rapidly?", "Qualitative / Subjective Temporal Query")
    ]

    # 3. Out-Of-Distribution (OOD) & Non-EO Queries
    ood_cases = [
        ("What will the temperature, cloudiness, and rainfall forecast be tomorrow in Delhi?", "Weather / Meteorology"),
        ("Write a Python script to compute the Fibonacci sequence up to N terms.", "Programming Assistance"),
        ("Who is the prime minister of India and when is the next national election?", "General Knowledge / Politics"),
        ("How do I fix a flat tire on my bicycle?", "Everyday Life Instruction"),
        ("Calculate the square root of 987654321.", "Pure Mathematics"),
        ("What are the best tourist attractions to visit in Paris during summertime?", "Travel Recommendations"),
        ("Can you draft a formal job application letter for a data scientist position?", "General Text Generation"),
        ("Explain Einstein's theory of general relativity in simple terms.", "Physics Theory"),
        ("Recommend five movies similar to Interstellar and Contact.", "Entertainment Recommendation"),
        ("How many calories are in a bowl of white rice with chicken curry?", "Diet & Nutrition")
    ]

    records = []

    # Evaluate In-Distribution
    id_correct = 0
    id_confidences = []
    id_review_count = 0

    print(f"\n1. Evaluating In-Distribution Paraphrases (N={len(in_distribution_cases)})...")
    for q, true_task in in_distribution_cases:
        route = router.predict(q)
        pred_task = route.selected_task
        is_corr = (pred_task == true_task)
        if is_corr:
            id_correct += 1
        id_confidences.append(route.confidence)
        if route.requires_review:
            id_review_count += 1

        records.append({
            "test_type": "in_distribution",
            "query": q,
            "true_category": true_task,
            "predicted_task": pred_task,
            "confidence": round(route.confidence, 4),
            "requires_review": route.requires_review,
            "correct": is_corr
        })

    id_accuracy = (id_correct / len(in_distribution_cases)) * 100.0
    id_mean_conf = float(np.mean(id_confidences))
    id_review_rate = (id_review_count / len(in_distribution_cases)) * 100.0

    print(f"  In-Distribution Accuracy : {id_correct}/{len(in_distribution_cases)} ({id_accuracy:.1f}%)")
    print(f"  In-Distribution Mean Conf: {id_mean_conf*100:.1f}%")
    print(f"  In-Distribution Review % : {id_review_rate:.1f}%")

    # Evaluate Ambiguous Cases
    amb_review_count = 0
    amb_confidences = []
    print(f"\n2. Evaluating Ambiguous & Multi-Intent Queries (N={len(ambiguous_cases)})...")
    for q, desc in ambiguous_cases:
        route = router.predict(q)
        amb_confidences.append(route.confidence)
        if route.requires_review:
            amb_review_count += 1

        records.append({
            "test_type": "ambiguous_multi_intent",
            "query": q,
            "true_category": desc,
            "predicted_task": route.selected_task,
            "confidence": round(route.confidence, 4),
            "requires_review": route.requires_review,
            "correct": None
        })

    amb_review_rate = (amb_review_count / len(ambiguous_cases)) * 100.0
    print(f"  Ambiguous Review Rate (Safety Flagged): {amb_review_count}/{len(ambiguous_cases)} ({amb_review_rate:.1f}%)")
    print(f"  Ambiguous Mean Confidence            : {np.mean(amb_confidences)*100:.1f}%")

    # Evaluate OOD Cases
    ood_review_count = 0
    ood_confidences = []
    print(f"\n3. Evaluating Out-Of-Distribution (OOD) Non-EO Queries (N={len(ood_cases)})...")
    for q, desc in ood_cases:
        route = router.predict(q)
        ood_confidences.append(route.confidence)
        if route.requires_review:
            ood_review_count += 1

        records.append({
            "test_type": "out_of_distribution",
            "query": q,
            "true_category": desc,
            "predicted_task": route.selected_task,
            "confidence": round(route.confidence, 4),
            "requires_review": route.requires_review,
            "correct": None
        })

    ood_review_rate = (ood_review_count / len(ood_cases)) * 100.0
    print(f"  OOD Review Rate (Safety Flagged)     : {ood_review_count}/{len(ood_cases)} ({ood_review_rate:.1f}%)")
    print(f"  OOD Mean Confidence                  : {np.mean(ood_confidences)*100:.1f}%")

    # Overall Safety Filter Effectiveness
    # Safety Filter's Job: Flag Ambiguous and OOD queries (< 0.65) while allowing high-confidence In-Distribution through
    total_safety_targets = len(ambiguous_cases) + len(ood_cases)
    safety_caught = amb_review_count + ood_review_count
    safety_recall = (safety_caught / total_safety_targets) * 100.0

    print("\n===========================================================================")
    print("                      SAFETY FILTER ROBUSTNESS METRICS")
    print("===========================================================================")
    print(f"In-Distribution Paraphrase Accuracy   : {id_accuracy:.2f}% ({id_correct}/{len(in_distribution_cases)})")
    print(f"Safety Filter Recall on Ambiguous/OOD : {safety_recall:.2f}% ({safety_caught}/{total_safety_targets} flagged for review)")
    print(f"OOD Interception Rate (< 0.65 conf)    : {ood_review_rate:.2f}%")
    print(f"Ambiguity Interception Rate (< 0.65)  : {amb_review_rate:.2f}%")
    print("===========================================================================")

    # Save to artifacts
    out_dir = REPO_ROOT / "artifacts" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_out = pd.DataFrame(records)
    csv_path = out_dir / "router_expanded_test_cases.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"\nSaved expanded test cases to: {csv_path}")

    summary = {
        "benchmark_title": "SatQuery Task Router Expanded Robustness & OOD Benchmark",
        "in_distribution": {
            "total": len(in_distribution_cases),
            "correct": id_correct,
            "accuracy_pct": round(id_accuracy, 2),
            "mean_confidence": round(id_mean_conf, 4),
            "review_rate_pct": round(id_review_rate, 2)
        },
        "ambiguous_multi_intent": {
            "total": len(ambiguous_cases),
            "flagged_for_review": amb_review_count,
            "review_rate_pct": round(amb_review_rate, 2),
            "mean_confidence": round(float(np.mean(amb_confidences)), 4)
        },
        "out_of_distribution": {
            "total": len(ood_cases),
            "flagged_for_review": ood_review_count,
            "review_rate_pct": round(ood_review_rate, 2),
            "mean_confidence": round(float(np.mean(ood_confidences)), 4)
        },
        "safety_filter_overall": {
            "safety_target_prompts": total_safety_targets,
            "safety_intercepted": safety_caught,
            "safety_interception_recall_pct": round(safety_recall, 2)
        }
    }

    json_path = out_dir / "router_expanded_robustness_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved robustness summary JSON to: {json_path}")


if __name__ == "__main__":
    run_expanded_evaluation()
