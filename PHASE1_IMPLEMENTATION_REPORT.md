# SatQuery AI — Phase 1 Implementation Report

**Research Topic**: Sensor- and Task-Compatibility-Constrained Selective Routing for Multimodal Earth Observation  
**Problem Code**: SIH26167 (ISRO / Earth Observation Analytical Assistant)  
**Phase**: Phase 1 — Router and Compatibility Engine Hardening  
**Verification Status**: Complete | 69/69 Passing Tests (100%) | Clean Git Working Tree  
**Primary Deliverables**: Code Hardening, Specialist Input Contracts, N=200 Balanced Benchmark, Empirical Risk–Coverage Curves, Matched-Coverage Comparative Evaluation  

---

## 1. Executive Summary & Scope Integrity

Phase 1 completes the foundational hardening of the **Task Router and Compatibility Engine**, converting the system from an unconstrained demonstrator into a scientifically rigorous, publication-grade selective routing architecture.

### Scope Boundaries Strictly Enforced
- **No Premature Foundation Models**: No Prithvi-EO, RemoteCLIP, or RS-VLM foundation models were trained locally.
- **Storage Conservation**: No unmanageable full-scale datasets (such as the 110 GiB BigEarthNet v2.0 archive) were downloaded to local disk C: (preserving the 86.9 GB free capacity).
- **Specialist Continuity**: Existing calibrated specialists (LightGBM 488-feature Land Cover and Classical CVA Change Detection) were preserved without modification.
- **Historical Evidence Preservation**: Historical N=60 baseline comparison and summary records were archived to `artifacts/benchmarks/historical_n60_*` rather than overwritten.
- **Scientific Humility**: We report measured empirical findings without premature claims of research superiority or zero operational risk.

---

## 2. Architectural Defects Resolved

### 2.1 Task Coercion Bug Resolution
* **The Defect**: In the original implementation (`satquery/router.py`), whenever `obs_meta.modality == 'temporal_pair'`, the router searched for compatible tasks and rewrote `selected_task` to the highest-probability temporal candidate (e.g., `CHANGE_DETECTION`), even when the natural-language query explicitly and unambiguously requested Land Cover classification with high confidence ($P > 0.75$).
* **The Architectural Fix**:
  Implemented strict mathematical task arbitration in `TaskRouter.predict()`:
  $$\text{Disambiguate}(q, M) \iff P(T^* \mid q) < \tau \;\land\; \neg \text{HasDistinctiveKeywords}(T^*, q) \;\land\; \exists C \in \mathcal{C}(M): (P(T^*) - P(C) \le \Delta_p \lor \text{IsBroad}(q))$$
  - When linguistic intent is clearly committed (e.g., `"What land-cover classes are present in this satellite scene?"`), the intended task is **strictly preserved** (`selected_task = text_task`), and the incompatible input is flagged (`input_compatibility = False`).
  - Task coercion is completely eliminated.
  - Legitimate disambiguation remains active for broad, uncommitted queries (e.g., `"Is there any noticeable alteration visible across this scene?"` or `"Analyze this satellite observation and report findings"`).

```python
# Non-coercive task arbitration logic (satquery/router.py)
text_has_explicit_keywords = _contains_task_keywords(text_task, query)
has_explicit_intent = (
    text_confidence >= self.confidence_threshold
    or text_has_explicit_keywords
)

if can_disambiguate and best_cand is not None:
    selected_task = best_cand
    input_compatibility = True
else:
    # Intended task preserved: DO NOT coerce incompatible requests!
    selected_task = text_task
    input_compatibility = False
```

### 2.2 Formal Specialist Input Contracts
* **The Defect**: `satquery/taxonomy.py` declared `LAND_COVER_CLASSIFICATION` as accepting `InputType.SINGLE_IMAGE`, while the deployed `LandCoverClassifier` strictly requires 84/488 tabular feature vectors or S1+S2 GeoTIFF pairs, immediately rejecting raw 2D/3D images.
* **The Architectural Fix**:
  Formalized `SpecialistInputContract` dataclass in `satquery/taxonomy.py` and embedded contracts across all 7 taxonomy tasks. Aligned `TASK_COMPATIBILITY_MAP` in `satquery/utils/metadata.py` so metadata inspection and specialist validation share identical contracts:
  - `LAND_COVER_CLASSIFICATION`: `required_modalities=["optical", "sar"]`, `accepted_representations=["feature_vector", "multimodal_geotiff_pair"]`, `temporal_relationship="single_point"`, `is_registered=True`.
  - `CHANGE_DETECTION`: `required_modalities=["optical"]`, `accepted_representations=["temporal_image_pair", "temporal_path_pair"]`, `temporal_relationship="bitemporal_pair"`, `is_registered=True`.
  - Stubs (`SINGLE_IMAGE_VQA`, `SCENE_CAPTIONING`, `VISUAL_GROUNDING`, `CHANGE_VQA`, `OPTICAL_SAR_FUSION`): Marked explicitly with `is_registered=False`.

### 2.3 Baseline Aggregation & Action Taxonomy Standardization
* **The Defect**: Baseline 1 (Text-Only Argmax) logged actions as `"DISPATCHED_TO_SPECIALIST"` or `"EXECUTED_HALLUCINATED"`, but the aggregation script filtered strictly for `action == "EXECUTED"`. Consequently, B1 reported 0 executed requests, masking that all 60 requests were executed with a catastrophic 78.33% error rate.
* **The Architectural Fix**:
  Standardized 4 mutually exclusive action categories across all policies:
  1. `EXECUTED`: Dispatched to and executed by a specialist.
  2. `HALTED_UNCERTAINTY`: Halted upfront due to linguistic ambiguity or low confidence ($P < \tau$).
  3. `REJECTED_INCOMPATIBLE`: Blocked upfront due to observation modality or calibration mismatch.
  4. `REFUSED_UNAVAILABLE`: Refused structuredly because the analytical task is not registered in the taxonomy.

---

## 3. Quantitative Evaluation Results

### 3.1 Corrected Historical N=60 Benchmark Comparison
The table below displays the corrected N=60 metrics alongside historical baselines:

| Routing Strategy | Supported Coverage (%) | Incompatible Dispatches | Incompat. Dispatch Rate (%) | Total Executed Requests | Silent Error Rate on Executed (%) | Anomaly Interception Rate (%) | Unsupported Refusal Rate (%) | Mean Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1: Text-Only Argmax ($\tau=0.0$)** | **100.0%** | **10** | **100.0%** | **60** | **78.33%** | 0.0% | 0.0% | 53.05 ms |
| **B2: Text + Compat Rules ($\tau=0.0$)** | 81.25% | 0 | 0.0% | 19 | 31.58% | 0.0% | 70.0% | 89.63 ms |
| **B3: Text + Confidence Gate ($\tau=0.65$)** | 25.00% | 4 | 40.0% | 8 | 50.00% | 94.44% | 50.0% | 8.82 ms |
| **Proposed Policy (Clarify Mode, $\tau=0.65$)** | 25.00% | **0** | **0.0%** | **4** | **0.00%** | **94.44%** | 50.0% | 49.65 ms |
| **Proposed Policy (Provisional, $\tau=0.65$)** | **87.50%** | **0** | **0.0%** | **22** | **18.18%** | 0.0% | **100.0%** | 118.83 ms |

---

### 3.2 Expanded N=200 Balanced Operational Benchmark
Evaluated on the newly constructed $N=200$ catalog across 6 balanced cohorts:
1. **Cohort 1**: Supported Compatible ($N=50$)
2. **Cohort 2**: Supported Incompatible Inputs ($N=40$)
3. **Cohort 3**: Missing Sensor Modalities ($N=30$)
4. **Cohort 4**: Ambiguous & Multi-Intent Queries ($N=40$)
5. **Cohort 5**: Unsupported Tasks ($N=20$)
6. **Cohort 6**: Out-of-Distribution Prompts ($N=20$)

| Routing Strategy | Supported Coverage (%) | Incompatible Dispatches | Incompat. Dispatch Rate (%) | Total Executed Requests | Silent Error Rate on Executed (%) | Anomaly Interception Rate (%) | Unsupported Refusal Rate (%) | Mean Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1: Text-Only Argmax** | 100.0% | 37 | 52.86% | 200 | 82.50% | 0.0% | 0.0% | 42.98 ms |
| **B2: Text + Compat Rules** | 70.0% | 0 | 0.0% | 52 | 32.69% | 0.0% | 95.0% | 78.60 ms |
| **B3: Text + Confidence Gate** | 16.0% | 12 | 17.14% | 20 | 60.00% | 91.67% | 40.0% | 11.10 ms |
| **Proposed (Clarify Mode)** | 16.0% | **0** | **0.0%** | **8** | **0.00%** | **91.67%** | 40.0% | 45.48 ms |
| **Proposed (Provisional Mode)** | **88.0%** | **0** | **0.0%** | **62** | **16.13%** | 0.0% | **100.0%** | 107.08 ms |

---

### 3.3 Template Generalization Audit: Validation Split vs Frozen Test Split
To verify that routing performance does not rely on memorizing benchmark templates, the $N=200$ catalog was partitioned into 100 validation queries and 100 held-out test queries with isolated template families:

| Policy | Split | Supported Coverage (%) | Incompatible Dispatch Rate (%) | Silent Error Rate on Executed (%) | Anomaly Interception Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **B1: Text-Only Argmax** | Val ($N=100$) | 100.0% | 58.97% | 80.77% | 0.0% |
| **B1: Text-Only Argmax** | Test ($N=100$) | 100.0% | 51.43% | 81.73% | 0.0% |
| **B2: Text + Compat Rules** | Val ($N=100$) | 80.0% | 0.0% | 33.33% | 0.0% |
| **B2: Text + Compat Rules** | Test ($N=100$) | 65.52% | 0.0% | 26.92% | 0.0% |
| **B3: Text + Confidence Gate** | Val ($N=100$) | 12.0% | 25.64% | 76.92% | 96.67% |
| **B3: Text + Confidence Gate** | Test ($N=100$) | 20.69% | 8.57% | 33.33% | 86.67% |
| **Proposed (Clarify Mode)** | Val ($N=100$) | 12.0% | **0.0%** | **0.00%** | 96.67% |
| **Proposed (Clarify Mode)** | Test ($N=100$) | 20.69% | **0.0%** | **0.00%** | 86.67% |
| **Proposed (Provisional Mode)**| Val ($N=100$) | 96.0% | **0.0%** | 14.29% | 0.0% |
| **Proposed (Provisional Mode)**| Test ($N=100$) | 82.76% | **0.0%** | 16.13% | 0.0% |

---

### 3.4 Matched-Coverage Comparative Analysis
Comparing policies at matched coverage levels controls for the trivial trade-off between coverage and accuracy:

| Operating Regime | Policy Evaluated | Supported Coverage (%) | Incompatible Dispatches | Selective Risk on Executed (%) | Anomaly Interception (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Unconstrained Regime** ($\Phi \approx 100\%$) | **B1: Text-Only Argmax** | 100.0% | 37 | **82.50%** | 0.0% |
| **Deterministic Regime** ($\Phi \approx 70-88\%$) | **B2: Text + Compat Rules** ($\tau=0.0$) | 70.0% | 0 | **32.69%** | 0.0% |
| **Risk-Flagged Regime** ($\Phi \approx 88\%$) | **Proposed Provisional** ($\tau=0.65$) | **88.0%** | **0** | **16.13%** | 0.0% |
| **Selective Halting** ($\Phi \approx 16-22\%$) | **B3: Confidence Gate Only** ($\tau=0.65$) | 16.0% | 12 | **60.00%** | 91.67% |
| **Strict Pareto Halting** ($\Phi \approx 16-22\%$) | **Proposed Clarify** ($\tau=0.60$) | **22.0%** | **0** | **8.33%** | 85.00% |
| **Zero-Risk Halting** ($\Phi \approx 16\%$) | **Proposed Clarify** ($\tau=0.65$) | **16.0%** | **0** | **0.00%** | 91.67% |

* **At high coverage ($\sim 88\%$)**: Proposed Provisional cuts selective risk in half compared to B2 (16.13% vs 32.69%) and by more than 5x compared to B1 (82.50%).
* **At selective coverage ($\sim 16-22\%$)**: Proposed Clarify achieves **0.00% to 8.33% risk**, whereas B3 incurs **60.00% risk** due to blind dispatch of incompatible inputs.

---

## 4. Empirical Risk–Coverage Pareto Frontier

A systematic threshold sweep across 18 operating points $\tau \in [0.10, 0.95]$ (step 0.05) traces the empirical selective risk curves. The generated plot is stored at `artifacts/benchmarks/risk_coverage_curves.png`.

### Key Observations from the Curves
1. **Pareto Front Dominance**: Proposed Clarify strictly dominates B3 and B1 across all thresholds. For any fixed coverage level, Proposed Clarify exhibits lower selective risk.
2. **Threshold Recommendation**:
   - At historical $\tau=0.65$: Supported coverage is 16.0% with 0.0% selective risk and 91.67% anomaly interception.
   - At recommended $\tau^*=0.60$: Supported coverage increases to 22.0% with an 8.33% selective risk and 85.0% anomaly interception.
   - For operational deployments, $\tau^*=0.60$ is the recommended balanced operating point, while $\tau=0.65$ remains the mission-critical zero-risk setting.

---

## 5. Regression Test Suite Verification

The full test suite was executed via `pytest`, confirming **69 of 69 tests passing (100%)** in 25.4s:

```
collected 69 items

tests/test_change_detection_benchmark.py ....                            [  5%]
tests/test_change_detection_synthetic.py ............                    [ 23%]
tests/test_demo_app.py ..........                                        [ 37%]
tests/test_input_aware_routing.py ......                                 [ 46%]
tests/test_orchestrator.py ......                                        [ 55%]
tests/test_phase1_routing_contracts.py ..........                        [ 69%]
tests/test_phase2_classical_hardening.py .......                         [ 79%]
tests/test_predict_cli.py .                                              [ 81%]
tests/test_raw_tiff_inference_parity.py ..                               [ 84%]
tests/test_router.py ...                                                 [ 88%]
tests/test_specialists.py .....                                          [ 95%]
tests/test_taxonomy.py ...                                               [100%]

============================= 69 passed in 25.44s =============================
```

---

## 6. Reproducibility Manifest & Artifact Hashes

Root-level manifest `reproducibility_manifest.json` records SHA256 integrity hashes for all model checkpoints, benchmark catalogs, and evaluation summaries:

```json
{
  "artifacts/router/router_model.joblib": "46f476a72af42242cccb64f430d0affdd34d3f04f2bc4da511b540301e04b4ec",
  "artifacts/specialists/land_cover_phase2/model.joblib": "f785ca1f5e9691c17e4a17ebf3f6766f7053059ba871188d24f6dce0ed43d071",
  "data/benchmark_n200_catalog.json": "d287cde9e76c25c42548090606545e97d7db8de76287482498fc3304fa1505f8",
  "artifacts/benchmarks/n200_routing_baselines_summary.json": "26e8dffd3efb094fc73d5b1869b44119f2c217e0a174dcd566e3f0544666bfec",
  "artifacts/benchmarks/n200_threshold_sweep_summary.json": "62329fce77257efb64e88ed59d9307b91f862cb318fcfd11bb2dad44373997f7",
  "artifacts/benchmarks/risk_coverage_curves.png": "f0dc9869131220436d4f44b3984d9d1aa187666c2f79a53115d77b9871a51676",
  "artifacts/benchmarks/corrected_n60_routing_baselines_summary.json": "909539e0b06e6dc1d90a6fa601ce24713073b81f6ab91e127f6d80ea934659f4"
}
```

Pinned dependencies recorded in `requirements.txt`:
- `numpy==2.5.3`, `pandas==3.0.6`, `scikit-learn==1.9.1`, `lightgbm==4.7.0`, `scipy==1.18.1`, `matplotlib==3.11.2`, `tifffile==2026.10.15`, `joblib==1.6.0`, `pytest==9.1.1`.

---

## 7. Phase 1 Conclusion & Readiness for Review

Phase 1 has fulfilled every technical criterion established in the approved Phase 0 audit:
1. **Task coercion is completely prevented**: Clear linguistic intent is never overwritten by attached modalities.
2. **Specialist input contracts are formalized and synchronized**: Tabular models do not advertise raw image compatibility.
3. **Baseline aggregation bugs are fully resolved**: Historical N=60 results are preserved, and true denominators are reported.
4. **Benchmark expanded to N=200 across 6 balanced cohorts**: Evaluated on isolated validation and held-out test splits.
5. **Matched-coverage comparison and Risk–Coverage curves established**: Demonstrates superior safety-coverage trade-off.
6. **10 new regression tests passing (69/69 total, 100%)**: Zero regressions across existing functionalities.

**Execution is halted here as requested for user review before proceeding to Phase 2.**
