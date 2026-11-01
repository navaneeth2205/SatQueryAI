# SatQuery AI — Research Upgrade: Phase 0 Repository Audit

**Repository**: `SatQueryAI`  
**Audit Target Commit**: [`297a7ee`](file:///c:/Users/B%20NAVANEETH/SatQueryAI) (`main` branch)  
**Date**: November 1, 2026  
**Auditor**: Lead Research Agent (Antigravity / Google DeepMind)  
**Research Objective**: Transitioning SatQuery AI from a hackathon proof-of-concept into a rigorous, publication-grade research framework focused on:  
> **"Sensor- and Task-Compatibility-Constrained Selective Routing for Multimodal Earth Observation"**

---

## 1. Executive Summary & Audit Purpose

This audit provides an unvarnished, line-by-line inspection of the SatQuery AI codebase, model checkpoints, data assets, and benchmark artifacts at commit `297a7ee`. The goal is to establish exact scientific baselines, mathematically verify all benchmark denominators, uncover architectural contradictions and bugs, and lay out an actionable, 4-week roadmap toward a defensible methodology submission.

### Key High-Level Findings
1. **Mathematical Artifact Discrepancies**: The reported 4-baseline routing comparison ($N=60$) was **not evaluated at matched coverage**. B2 had 87.5% coverage (14/16), Proposed (Provisional) had 81.25% coverage (13/16), and Proposed (Clarify) had only 25.0% coverage (4/16). Furthermore, an aggregation script bug coded B1 actions as `"DISPATCHED_TO_SPECIALIST"` rather than `"EXECUTED"`, causing summary tables to erroneously report 0 total executed requests for B1.
2. **The "Zero Error" Denominator Trap**: In Clarify mode, the reported 0.0% error rate was achieved because **only 5 out of 60 requests were executed** (4 supported requests + 1 misrouted unsupported task). The remaining 55 requests were halted or refused. In Provisional mode, 21 requests executed, suffering **5 errors (23.8% error rate)**.
3. **Task Coercion Bug in Router**: In [`satquery/router.py`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/satquery/router.py#L208-L228), attaching a temporal pair forces the router to coerce the task to `CHANGE_DETECTION`, even when the user explicitly asked for land-cover classification or optical-SAR fusion. This caused land-cover queries to be erroneously executed as change detection (e.g., scenario 261).
4. **Compatibility Map vs. Specialist Contradiction**: [`satquery/utils/metadata.py`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/satquery/utils/metadata.py#L20-L25) declares `SINGLE_IMAGE` compatible with `LAND_COVER_CLASSIFICATION`. However, the actual specialist in [`satquery/specialists/land_cover.py`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/satquery/specialists/land_cover.py#L115-L121) **strictly rejects any 2D or 3D single image**, demanding precomputed 84/488 feature vectors or an explicit S1+S2 GeoTIFF dictionary.
5. **Compute & Storage Constraints**: Free disk space on the development workstation (`C:`) is **86.9 GB**. Downloading the full BigEarthNet v2.0 dataset (110 GiB) will overflow local storage. The active Python runtime is **Python 3.14.3**, which currently lacks official CUDA PyTorch binaries, and the local GPU is an **NVIDIA RTX 2050 (4 GB VRAM)**. Deep learning foundation model training (Prithvi-EO, RS-VLM) must be run on cloud GPUs.

---

## 2. Current Architecture & Pipeline Flow

The current implementation follows a decoupled 4-stage pipeline:

```mermaid
flowchart TD
    Q["Natural Language Query (q)"] --> Router["TaskRouter\n(TF-IDF + Calibrated Logistic Regression)"]
    Router --> Intent["Linguistic Probabilities P(t | q)\nTop Intent: text_task, text_confidence"]
    
    Data["Raw Satellite Input (x)"] --> Meta["Metadata Extractor\n(extract_observation_metadata)"]
    Meta --> ObsMeta["InputObservationMetadata (z)\n(modality, sensors, dtypes, shapes)"]
    
    Intent --> Arb["Compatibility-Aware Selector\n(satquery/router.py L191-L247)"]
    ObsMeta --> Arb
    
    Arb --> Route["ExecutionRoute\n(selected_task, confidence, requires_review)"]
    Route --> Gate["Selective Decision Policy Gate\n(satquery/orchestrator.py L132-L185)"]
    
    Gate -->|"conf < 0.65 & clarify"| Clarify["NEEDS_CLARIFICATION\n(Halt Execution)"]
    Gate -->|"conf >= 0.65 or provisional"| Registry["SpecialistRegistry.dispatch()\n(satquery/specialists/registry.py)"]
    
    Registry -->|"Specialist None"| Stub["UNAVAILABLE_SPECIALIST\n(Honest Refusal Stub)"]
    Registry -->|"Incompatible Input"| Incompat["INCOMPATIBLE_INPUT\n(Pre-execution Validation)"]
    Registry -->|"Compatible & Implemented"| Specialist["Execute Specialist\n(LandCover 488 / CVA Change)"]
    
    Specialist --> Evidence["EvidenceEngine & Execution Trace\n(satquery/evidence.py)"]
```

### Detailed Component Implementation Audit

| Component | Source File & Line Range | Exact Algorithm / Model | Key Configuration / Parameters |
| :--- | :--- | :--- | :--- |
| **Query Intent Encoder** | [`satquery/router.py#L90-L139`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/satquery/router.py#L90-L139) | `TfidfVectorizer` (sublinear TF, n-grams 1-2, English stop words) + `LogisticRegression` (balanced class weights, L-BFGS, C=1.0) | Fitted on 109 starter queries across 7 taxonomy classes (`configs/router_config.json`). |
| **Probability Calibration** | [`satquery/router.py#L108-L112`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/satquery/router.py#L108-L112) | Platt Sigmoid Scaling via `CalibratedClassifierCV(method="sigmoid", cv=min(3, min_samples))` | Post-hoc calibrated probability vector $P(t_k \mid q)$. |
| **Metadata Extraction** | [`satquery/utils/metadata.py#L91-L233`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/satquery/utils/metadata.py#L91-L233) | Rule-based array dimension and dictionary key inspector | Distinguishes `feature_vector` (1D), `single_image` (2D/3D), `temporal_pair` (list/tuple of 2 or t1/t2 dict), and `optical_sar_pair` (s1/s2 dict). |
| **Compatibility Arbitration** | [`satquery/router.py#L191-L247`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/satquery/router.py#L191-L247) | Lookup against `TASK_COMPATIBILITY_MAP` with temporal disambiguation | **Bug Present**: If temporal pair is attached, overrides `text_task` with top compatible temporal candidate. |
| **Selective Decision Gate** | [`satquery/orchestrator.py#L132-L185`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/satquery/orchestrator.py#L132-L185) | Dual policy arbitration: `clarify` vs `provisional` with threshold $\tau=0.65$ | If $P(t \mid q) < 0.65$: `clarify` policy halts execution; `provisional` policy executes with warning disclaimers. |
| **Specialist Registry** | [`satquery/specialists/registry.py#L20-L77`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/satquery/specialists/registry.py#L20-L77) | In-memory dictionary `_specialists: Dict[TaskType, BaseSpecialist]` | Only 2 specialists registered. Dispatches registered models or returns structured `SpecialistStatus.UNAVAILABLE` stubs. |

---

## 3. Implemented versus Stubbed Capabilities

The active taxonomy defines 7 tasks ([`satquery/taxonomy.py#L14-L63`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/satquery/taxonomy.py#L14-L63)). The table below reports their actual implementation state in the code:

| Task Type | Taxonomy Key | Model Family in Registry | Actual Implementation Status | Exact Source Reference |
| :--- | :--- | :--- | :--- | :--- |
| **1. Land Cover Classification** | `LAND_COVER_CLASSIFICATION` | `LandCoverClassifier` | **Fully Implemented**: LightGBM (16 binary One-vs-Rest models) on 488 multimodal features + per-class decision thresholds. Also supports Random Forest baseline (84 features). | [`satquery/specialists/land_cover.py#L32-L274`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/satquery/specialists/land_cover.py#L32-L274) |
| **2. Change Detection** | `CHANGE_DETECTION` | `TemporalDifferenceBaseline` | **Fully Implemented (Unsupervised)**: Normalized CVA with Gaussian smoothing ($\sigma=1.2$) and Dual-Threshold Hysteresis spatial expansion. No learned deep weights. | [`satquery/specialists/change_detection.py#L35-L378`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/satquery/specialists/change_detection.py#L35-L378) |
| **3. Optical-SAR Fusion** | `OPTICAL_SAR_FUSION` | None (Unregistered) | **Split Status**: Internal feature-level fusion is operational in Land Cover (`satquery/features/multimodal.py`); standalone fusion task is an **Unregistered Honest Stub**. | [`satquery/specialists/registry.py#L45-L60`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/satquery/specialists/registry.py#L45-L60) |
| **4. Single-Image VQA** | `SINGLE_IMAGE_VQA` | None (Unregistered) | **Honest Stub**: Returns `UNAVAILABLE_SPECIALIST` explaining roadmap milestone. | [`satquery/specialists/registry.py#L45-L60`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/satquery/specialists/registry.py#L45-L60) |
| **5. Change VQA** | `CHANGE_VQA` | None (Unregistered) | **Honest Stub**: Returns `UNAVAILABLE_SPECIALIST` explaining roadmap milestone. | [`satquery/specialists/registry.py#L45-L60`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/satquery/specialists/registry.py#L45-L60) |
| **6. Scene Captioning** | `SCENE_CAPTIONING` | None (Unregistered) | **Honest Stub**: Returns `UNAVAILABLE_SPECIALIST` explaining roadmap milestone. | [`satquery/specialists/registry.py#L45-L60`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/satquery/specialists/registry.py#L45-L60) |
| **7. Visual Grounding** | `VISUAL_GROUNDING` | None (Unregistered) | **Honest Stub**: Returns `UNAVAILABLE_SPECIALIST` explaining roadmap milestone. | [`satquery/specialists/registry.py#L45-L60`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/satquery/specialists/registry.py#L45-L60) |

---

## 4. Benchmark Denominator and Metric Audit ($N=60$)

We audited [`scripts/benchmark_routing_policy_baselines.py`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/scripts/benchmark_routing_policy_baselines.py) and the raw record artifact [`artifacts/benchmarks/routing_baselines_comparison.csv`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/artifacts/benchmarks/routing_baselines_comparison.csv).

### Cohort Distribution ($N=60$ scenarios)
* **Cohort 1**: Supported In-Domain with Compatible Inputs ($N=16$)
* **Cohort 2**: Supported In-Domain with Incompatible Inputs ($N=10$)
* **Cohort 3**: Multimodal Disambiguation Cases ($N=6$)
* **Cohort 4**: Unsupported Analytical Tasks ($N=10$)
* **Cohort 5**: Linguistically Ambiguous & Multi-Intent Queries ($N=9$)
* **Cohort 6**: Out-Of-Distribution (Non-EO) Queries ($N=9$)

### Mathematical Verification of Raw Outcomes per Baseline

```
========================================================================================================================
Cohort & Scenario Counts             B1: Text-Only      B2: Text + Rules    B3: Text + Conf     Proposed (Clarify)  Proposed (Provisional)
========================================================================================================================
Cohort 1 (N=16 Supported Compat)
  - Executed Correctly               13                 13                  4                   4                   13
  - Executed Incorrectly (Errors)     1                  1                  0                   0                    0
  - Halted (Needs Clarification)      0                  0                 12                  12                    0
  - Refused / Hallucinated            2 (Hallucinated)   0                  0                   0                    3 (Refused)
  - Rejected Incompatible             0                  2                  0                   0                    0
------------------------------------------------------------------------------------------------------------------------
Cohort 2 (N=10 Supported Incompat)
  - Incompatible Dispatches          10                  0                  4                   0                    0
  - Rejected Incompatible             0                  8                  0                   3                    7
  - Halted (Needs Clarification)      0                  0                  6                   6                    0
  - Refused (Unavailable)             0                  0                  0                   1                    2
  - Erroneously Executed              0                  2                  4                   0                    1 (Coerced to CD!)
------------------------------------------------------------------------------------------------------------------------
Cohort 3 (N=6 Disambiguation)
  - Executed Correctly                0                  1                  0                   0                    2
  - Halted (Needs Clarification)      0                  0                  6                   6                    0
  - Rejected Incompatible             0                  5                  0                   0                    0
  - Refused / Hallucinated            6 (Hallucinated)   0                  0                   0                    4 (Refused)
------------------------------------------------------------------------------------------------------------------------
Cohort 4 (N=10 Unsupported Tasks)
  - Explicit Refusal (Unavailable)    0                  7                  5                   4                    9
  - Hallucinated Execution           10                  0                  0                   0                    0
  - Halted (Needs Clarification)      0                  0                  5                   5                    0
  - Erroneously Executed              0                  0                  0                   1 (Coerced to CD!)   1 (Coerced to CD!)
  - Rejected Incompatible             0                  3                  0                   0                    0
------------------------------------------------------------------------------------------------------------------------
Cohorts 5 & 6 (N=18 Anomaly & OOD)
  - Intercepted / Halted (Safe)       0 / 18             0 / 18            17 / 18 (94.4%)     17 / 18 (94.4%)       0 / 18
  - Erroneously Executed / Slipped   18 / 18            13 / 18             1 / 18              0 / 18               4 / 18 (Prov. CD)
  - Refused / Rejected                0                  5                  0                   1                   14
========================================================================================================================
OVERALL TOTALS (N=60)
  - Total Requests Executed          60                 37                 14                   5                   21
  - Total Executed Errors            47                 14                  5                   0                    5
  - Executed Error Rate (%)          78.3%              37.8%              35.7%                0.0%                23.8%
  - Supported Coverage (Cohort 1)    81.3% (13/16)      81.3% (13/16)      25.0% (4/16)        25.0% (4/16)         81.3% (13/16)
========================================================================================================================
```

### Critical Findings & Mathematical Bugs Uncovered in Audit

1. **Aggregation Script Bug for Baseline 1**:
   In `evaluate_baseline_1_text_only` ([`scripts/benchmark_routing_policy_baselines.py#L370-L385`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/scripts/benchmark_routing_policy_baselines.py#L370-L385)), the action was logged as `"DISPATCHED_TO_SPECIALIST"` or `"EXECUTED_HALLUCINATED"`. The summary aggregation function ([`#L742`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/scripts/benchmark_routing_policy_baselines.py#L742)) strictly looked for `action == "EXECUTED"`. Consequently, B1's total executed requests was printed as **0**, supported coverage as **0.0%**, and executed error rate as **0.0%** in [`artifacts/benchmarks/routing_baselines_summary.json`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/artifacts/benchmarks/routing_baselines_summary.json#L2-L13). In reality, B1 executed all **60/60** requests blindly, suffering **47 errors (78.3% error rate)**.
2. **The "Matched Coverage" Claim is Invalid**:
   The report stated: *"The proposed policy achieves a superior trade-off at matched coverage."*
   However, the actual measured coverage on Cohort 1 was:
   * B2: **81.25%** (13/16 correct executions; 2 rejected incompatible, 1 error)
   * Proposed (Provisional): **81.25%** (13/16 correct executions; 3 refused)
   * Proposed (Clarify): **25.00%** (4/16 correct executions; 12 halted)
   * B3: **25.00%** (4/16 correct executions; 12 halted)  
   Comparing Proposed (Clarify, 25% coverage) directly against B2 (81.25% coverage) and claiming superior error rate is mathematically invalid because the two operate at vastly different points along the risk-coverage curve.
3. **The "0.0% Error" Denominator Limitation**:
   Proposed (Clarify) achieved 0.0% error rate solely because it halted 55 out of 60 requests. Only 5 requests were allowed to execute. When the policy is relaxed to Provisional mode to achieve competitive coverage (81.25%), the error rate rises to **23.8% (5 errors / 21 executed)** due to task coercion on ambiguous queries and temporal inputs.
4. **The False Pareto-Optimality Claim for $\tau=0.65$**:
   In [`artifacts/benchmarks/threshold_sweep_analysis.csv`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/artifacts/benchmarks/threshold_sweep_analysis.csv), the measured outcomes across thresholds are:
   * $\tau = 0.60$: Clarify Coverage = **31.25%** (5/16), Provisional Coverage = 81.25%, Executed Count = 6, Errors = 0 (**0.0% Error**), Anomaly Interception = **94.44%**.
   * $\tau = 0.65$: Clarify Coverage = **25.00%** (4/16), Provisional Coverage = 81.25%, Executed Count = 5, Errors = 0 (**0.0% Error**), Anomaly Interception = **94.44%**.  
   At $\tau=0.60$, the system achieved higher coverage than at $\tau=0.65$ with identical zero error and identical anomaly interception. Therefore, claiming that 0.65 is mathematically Pareto-optimal over 0.60 was incorrect.

---

## 5. Dataset Readiness Audit

| Dataset | Current Local State | Required Canonical Version | Format & Annotations | Licensing & Access | Storage / Compute Requirement | Action Required for Migration |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BigEarthNet** | Local Serbia subset only: 14k patches as `.npy` matrices (`data/X_train_phase2.npy`, etc.). Only 4 raw GeoTIFF patches present. | **BigEarthNet v2.0 (reBEN)**: 549,488 paired Sentinel-1/Sentinel-2 patches across Europe. | Multi-band GeoTIFFs (12 S2 bands, 2 S1 bands), multi-label CORINE (19/43 classes) + pixel segmentation maps. | Open access (CDLA-Permissive-1.0). Available via official portal and Hugging Face. | **51 GiB (S1) + 59 GiB (S2) = 110 GiB compressed** (~160 GiB uncompressed). Exceeds local C: drive free space (86.9 GB). | Build a streaming / shard-based dataloader or sample a representative multi-country benchmark subset (20k patches, ~12 GB) for local development; full training on cloud storage. |
| **BigEarthNet.txt** | **0 bytes in repository**. | **BigEarthNet.txt Benchmark**: 1,082 verified pairs, 15,029 verified annotations. Full: 464,044 pairs with 9.6M text annotations. | JSON / Parquet text prompts: Captions, visual question-answering pairs, and referring expressions. | Open access. Hosted on Hugging Face (`BIFOLD-BigEarthNetv2-0/BigEarthNet.txt`). | Benchmark text subset: **< 150 MB**. Full text dataset: ~2.5 GB. | Download the 1,082-pair verified benchmark subset. Use it to expand router prompts and create an authentic remote-sensing VQA evaluation split. |
| **OSCD** | **Fully Present Locally**: 24 Sentinel-2 bi-temporal scenes in `data/oscd/` (14 train, 10 test). Cataloged in `data/change_detection_catalog.parquet`. | **Onera Satellite Change Detection**: Official 24-scene benchmark. | Multi-band TIFF and PNG RGB channels, binary change raster masks. | Academic open access (Daudt et al.). | Already downloaded and organized locally (~2.1 GB). | Retain the official 10-scene test set (`pisa`, `mumbai`, `beirut`, etc.) as strictly held-out. Reserve 4 scenes from the 14 training pairs for validation. |
| **RSVQA** | **0 bytes in repository**. | **RSVQA-LR** (Low Resolution / Sentinel-2) & **RSVQA-HR** (High Resolution / Aerial). | GeoTIFF images with JSON question-answer-type tuples (presence, counting, comparison). | Academic open access (Lobry et al., Université de Genève). | RSVQA-LR: ~1.8 GB. | Download RSVQA-LR (Sentinel-2 compatible). Use queries to stress-test the router and evaluate future VLM specialists. |
| **LEVIR-CD** | **0 bytes in repository**. | **LEVIR-CD**: 637 bi-temporal image pairs ($1024 \times 1024$, 0.5m VHR). | Optical RGB pairs with binary building change masks. | Academic open access (Chen & Shi, Beihang University). | ~4.2 GB compressed. | Ingest sample pairs to test router resolution and sensor metadata checks (verifying that VHR aerial imagery is distinguished from 10m Sentinel-2). |
| **CDVQA** | **0 bytes in repository**. | **CDVQA**: Bi-temporal remote-sensing image question answering. | Temporal image pairs with question-answer annotations describing changes. | Academic research repository. | ~1.5 GB. | Planned for Phase 3 when temporal vision-language specialist is introduced. |

---

## 6. Model Readiness Audit

```mermaid
flowchart LR
    subgraph Fully_Operational["Fully Operational in Pipeline"]
        LC["Land Cover (LightGBM 488)\nMicro F1: 0.7380\nMacro F1: 0.4615"]
        CD["Change Detection (CVA)\nSmoothed F1: 0.3393\nHysteresis Recall: 0.4430"]
        Router["Task Router\nCalibrated Logistic Reg\nAccuracy: 81.3%"]
    end
    
    subgraph Roadmap_Stubs["Honest Refusal Stubs (UNAVAILABLE_SPECIALIST)"]
        VQA1["Single-Image VQA"]
        VQA2["Change VQA"]
        Cap["Scene Captioning"]
        Grd["Visual Grounding"]
        Fus["Standalone SAR-Optical Fusion"]
    end
```

### Readiness Assessment by Capability
1. **Optical Land-Cover Classification**: **Partially Ready**. The model (`LandCoverClassifier_LightGBM_Phase2`) performs well on pre-extracted 488 features, but cannot directly accept raw optical-only images. It demands both S1 SAR and S2 optical GeoTIFFs.
2. **SAR Land-Cover Classification**: **Not Implemented**. SAR alone cannot be classified by the existing model.
3. **Optical–SAR Feature Fusion**: **Internally Operational**. Polarimetric and multispectral indices are fused into the 488 tabular feature vector. Standalone image-level fusion is a refusal stub.
4. **Bi-Temporal Change Detection**: **Baseline Ready**. Classical CVA runs deterministically with zero learnable weights. A learned temporal deep model (e.g., ChangeFormer or TinyCD) is required to advance beyond unsupervised baselines.
5. **Single-Image VQA & Change VQA**: **Stubbed**. No generative VLM weights are loaded. Deterministic refusal stubs prevent hallucination.
6. **Scene Captioning & Visual Grounding**: **Stubbed**. Handled via explicit refusal stubs.

---

## 7. Versioning & Reproducibility Gaps

| Dimension | Current State in Repository | Missing Requirement for Publication Grade | Required Remediation |
| :--- | :--- | :--- | :--- |
| **Dependency Pinning** | `pyproject.toml` contains abstract minimum bounds (`scikit-learn>=1.3.0`). **No `requirements.txt` or lockfile exists**. | Exact, bit-for-bit reproducible environment lockfile. | Generate a frozen `requirements.txt` with exact pip hashes and a Poetry/Pip-tools lockfile. |
| **Runtime Compatibility** | Active environment is **Python 3.14.3 on Windows**. | PyTorch, torchvision, and CUDA drivers require **Python 3.10–3.12** for stable wheels. | Set up a parallel Python 3.11 virtual environment for deep learning and foundation model training. |
| **Model Provenance** | `reproducibility_manifest.json` tracks script hashes and hyperparameters, but **omits SHA-256 hashes of the `.joblib` model binary files**. | Cryptographic model binary verification. | Recompute and add SHA-256 hashes of `model.joblib` for both router and land-cover models into the manifest. |
| **Dataset Tracking** | No DVC or dataset version tracking. Data files reside loosely in `data/`. | DVC (`.dvc` files and `dvc.yaml`) tracking remote storage and data checksums. | Initialize DVC, track `data/` assets, and record remote bucket storage pointers. |
| **Experiment Tracking** | Results are saved manually as JSON/CSV in `artifacts/benchmarks/`. | Centralized experiment logging with MLflow. | Integrate MLflow logging for parameter sweeps, metrics, and artifact tracking. |

---

## 8. Architectural Risks, Contradictions, and Bugs

### Risk 1: The Forced Task Coercion Bug
* **Location**: [`satquery/router.py#L208-L228`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/satquery/router.py#L208-L228)
* **Code Trace**:
  ```python
  if obs_meta.modality == InputType.TEMPORAL_PAIR.value:
      temporal_candidates = [cls for cls in self.classes_ if obs_meta.is_compatible_with(cls)]
      if temporal_candidates:
          cand_probs = [prob_dict[cls] for cls in temporal_candidates]
          best_cand_idx = int(np.argmax(cand_probs))
          selected_task = temporal_candidates[best_cand_idx]
  ```
* **Bug**: If a user submits a land-cover classification query ("Catalog biophysical surface categories in this footprint") but inadvertently attaches a temporal pair, the router checks which tasks are compatible with a temporal pair (`CHANGE_DETECTION` and `CHANGE_VQA`). It then selects `CHANGE_DETECTION` and executes change analysis!
* **Scientific Violation**: The router should NEVER rewrite a clear linguistic intent into an entirely different task just because the attached input happens to be temporal. It must preserve the user's intent, flag an `INCOMPATIBLE_INPUT` error, and instruct the user to attach a single image.

### Risk 2: Contradiction Between Metadata Map and Specialist Validation
* **Location**: [`satquery/utils/metadata.py#L22`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/satquery/utils/metadata.py#L22) vs [`satquery/specialists/land_cover.py#L121`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/satquery/specialists/land_cover.py#L121)
* **Issue**: `TASK_COMPATIBILITY_MAP` declares that `SINGLE_IMAGE` is compatible with `LAND_COVER_CLASSIFICATION`. However, if a user actually attaches a single 2D or 3D numpy image array (e.g., shape `[256, 256, 3]`), `LandCoverClassifier.supports_input()` rejects it with: `"Expected 84 features (or 488 Phase 2 features), got array of shape (256, 256, 3)"`.
* **Fix**: Provide a dedicated single-image preprocessing wrapper in the land cover specialist, or update the taxonomy to explicitly specify `multimodal_geotiff_pair` vs `feature_vector`.

### Risk 3: Small Benchmark Size and Sparse Execution
* The current 60 scenarios provide only 16 supported compatible cases, 10 incompatible cases, and 18 anomaly cases.
* In Clarify mode, evaluating error rates on a sample size of **$N=5$ executed requests** provides insufficient statistical power (a single error shifts the error rate by 20%). The benchmark must be expanded to at least **$N=200+$ scenarios** with bootstrap confidence intervals.

---

## 9. Prioritized Implementation Plan (4-Week Roadmap)

### Phase 1 (Week 1): Router & Compatibility Engine Hardening
* [ ] **Fix Task Coercion**: Eliminate forced task rewriting in [`satquery/router.py`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/satquery/router.py). When an incompatible modality is attached, preserve `selected_task = text_task` and set `input_compatibility = False` so that an informative `INCOMPATIBLE_INPUT` remediation is returned.
* [ ] **Harmonize Input Validation**: Update `LandCoverClassifier` to gracefully accept raw optical images by integrating a single-image feature extraction fallback.
* [ ] **Expand Routing Benchmark ($N=200$)**: Build a diverse benchmark with balanced cohorts:
  * 50 Supported Compatible requests across all 7 tasks
  * 40 Incompatible Input requests (dimension mismatches, missing sensors, temporal mismatches)
  * 30 Missing Modality requests (Optical without SAR, SAR without Optical)
  * 40 Ambiguous & Multi-intent requests
  * 40 Out-Of-Distribution non-EO prompts
* [ ] **Generate True Risk-Coverage Curves**: Implement a sweep over $\tau \in [0.10, 0.95]$ with step $0.05$ to plot empirical Risk-Coverage curves and compare B1, B2, B3, and Proposed at **identical coverage levels**.

### Phase 2 (Week 2): Data Ingestion & Manifest Management
* [ ] **BigEarthNet.txt Benchmark**: Ingest the 1,082 verified image-text pairs from Hugging Face.
* [ ] **reBEN Representative Subset**: Download a geographically stratified sample (20k patches, ~12 GB) of BigEarthNet v2.0 covering distinct European biomes.
* [ ] **LEVIR-CD & RSVQA-LR Samples**: Ingest sample sets for high-resolution optical change detection and Sentinel-2 VQA.
* [ ] **DVC Integration**: Initialize DVC and generate cryptographic manifests for all data subsets.

### Phase 3 (Week 3): Specialist Upgrades & Foundation Models
* [ ] **Single-Sensor Land Cover Models**: Train standalone Optical-only (S2 70-feat) and SAR-only (S1 14-feat) LightGBM models to handle missing-modality scenarios.
* [ ] **Learned Change Detection Baseline**: Implement a lightweight deep change detection model (e.g., TinyCD / Siamese ResNet) on OSCD and LEVIR-CD to benchmark against classical CVA.
* [ ] **Remote-Sensing VLM Integration**: Connect an open-source lightweight vision-language model (e.g., PaliGemma-3B or GeoChat via API/cloud worker) to replace VQA refusal stubs.

### Phase 4 (Week 4): Large-Scale Experiments & Methodology Submission
* [ ] **Sensor Missingness Stress Test**: Evaluate routing accuracy when S1 or S2 is systematically dropped.
* [ ] **Geographic Generalization**: Evaluate land-cover classification on held-out tiles outside Serbia.
* [ ] **Statistical Significance**: Compute 95% bootstrap confidence intervals for all routing metrics.
* [ ] **Final Methodology Document**: Compile the complete research paper methodology with empirical risk-coverage plots, ablation tables, and failure taxonomy.

---

## 10. Exact Files That Must Be Changed

```
SatQueryAI/
├── satquery/
│   ├── router.py                  # L208-228: Remove task coercion bug; fix ExecutionRoute scoring
│   ├── orchestrator.py            # L132-185: Refine selective decision actions; integrate risk-coverage logging
│   ├── taxonomy.py                # L20-45: Clarify input requirements (multimodal pair vs single image)
│   ├── utils/
│   │   └── metadata.py            # L20-45: Align TASK_COMPATIBILITY_MAP with specialist requirements
│   └── specialists/
│       ├── land_cover.py          # L107-140: Add single-sensor fallback (optical-only, SAR-only)
│       └── registry.py            # L40-75: Standardize refusal messages with structured roadmap metadata
├── scripts/
│   ├── benchmark_routing_policy_baselines.py  # L370-385, L740-760: Fix B1 execution logging; calculate matched-coverage curves
│   └── evaluate_expanded_router_robustness.py # Expand prompt suites to N=200
├── configs/
│   └── router_config.json         # Add multi-threshold configurations and calibration settings
├── requirements.txt               # [NEW FILE]: Pinned production dependencies
└── dvc.yaml                       # [NEW FILE]: Dataset and pipeline stage versioning
```

---

## 11. Estimated Compute and Storage Requirements

### Workstation Storage Budget
* **Current Free Space on `C:`**: `86.9 GB`
* **Local Allocation Plan**:
  * OSCD Dataset: `2.1 GB` (already present)
  * BigEarthNet Serbia Subset: `3.5 GB` (already present)
  * BigEarthNet.txt Benchmark Subset: `0.5 GB`
  * reBEN Stratified Subset (20k patches): `12.0 GB`
  * RSVQA-LR Sample: `1.8 GB`
  * Model Checkpoints & Virtualenvs: `8.0 GB`
  * **Total Planned Local Usage**: **~28 GB** (well within the 86.9 GB limit, leaving >55 GB headroom).

### Compute & Hardware Strategy
* **Local Laptop (Intel CPU + NVIDIA RTX 2050 4GB VRAM)**:
  * Dedicated to: Task Router training, tabular LightGBM training (488 features), CVA change detection, routing policy benchmarks, unit tests, and Flask web serving.
  * Runtime: Python 3.11 virtual environment.
* **Cloud GPU (Google Colab / RunPod / Lambda GPU with A10G / A100)**:
  * Dedicated to: Prithvi-EO-2.0 fine-tuning, PaliGemma / GeoChat VLM inference, and full BigEarthNet v2.0 pre-training.
  * Storage: Ephemeral cloud volume or mounted Google Drive / S3 bucket.

---

## 12. Conclusion & Phase 0 Gate Sign-off

The repository audit confirms that SatQuery AI possesses a solid engineering foundation (59/59 passing tests, fast deterministic CVA, and high-performance LightGBM land-cover classification). However, the routing novelty claims require hardening:
1. The **forced task coercion bug** must be removed immediately.
2. The benchmark must be scaled to **$N=200$** scenarios.
3. The routing policies must be compared across **full risk-coverage curves**, not unaligned single-point thresholds.

**Phase 0 Audit Status: COMPLETE.**  
Ready to proceed to Phase 1 (Router & Compatibility Engine Hardening) upon approval.
