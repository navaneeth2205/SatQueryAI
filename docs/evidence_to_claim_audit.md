# SatQuery AI: Evidence-to-Claim Audit Table

This document provides a comprehensive traceability matrix connecting every major experimental claim and metric in the **SatQuery AI Research Methodology Report** (`METHODOLOGY.md`) directly to saved repository artifacts, test arrays, evaluation scripts, and commit hashes.

---

## 1. Master Evidence-to-Claim Traceability Matrix

| Claim # | Experimental Claim in Methodology | Reported Value / Finding | Authoritative Supporting Repository Artifact | Verification Script or Test | Git Commit Reference | Audit Status |
| :-: | :--- | :--- | :--- | :--- | :---: | :---: |
| **C1** | **M3.1 Land-Cover ASL Macro F1** | Standard: **45.53%**<br>Test-Present: **52.03%** | `artifacts/specialists/deep_land_cover_asl/evaluation_summary.json`<br>`artifacts/specialists/m31_per_class_audit.csv` | `tests/test_m31_scorecard_regime_consistency.py` | `cb167b7` | **AUDITED & VERIFIED** |
| **C2** | **M3.1 Land-Cover ASL Micro F1** | **74.47%** | `artifacts/specialists/deep_land_cover_asl/evaluation_summary.json` | `tests/test_m31_scorecard_regime_consistency.py` | `cb167b7` | **AUDITED & VERIFIED** |
| **C3** | **M3.1 Land-Cover CB-BCE Hamming Loss** | **0.0816** (Lowest label error rate) | `artifacts/specialists/deep_land_cover_cb_bce/evaluation_summary.json` | `tests/test_m31_scorecard_regime_consistency.py` | `cb167b7` | **AUDITED & VERIFIED** |
| **C4** | **M3.1 LightGBM Reconciliation** | Standard: **41.14%**<br>Present: **47.02%**<br>(47.39% was legacy `zero_division=1.0`) | `artifacts/benchmarks/land_cover_reconciliation_audit.json`<br>`artifacts/benchmarks/land_cover_per_class_test_metrics.csv` | `tests/test_m31_scorecard_regime_consistency.py` | `ce71342` | **AUDITED & VERIFIED** |
| **C5** | **BigEarthNet S2 Preprocessing** | Archival (PB < 04.00): $\text{DN}/10000.0$<br>Modern (PB $\ge$ 04.00): $(\text{DN}-1000)/10000.0$ | `satquery/specialists/dataset_land_cover.py`<br>`PHASE2_BASELINE_VERIFICATION_AUDIT.md` | `scripts/compute_normalization_statistics.py` | `ce71342` | **AUDITED & VERIFIED** |
| **C6** | **BigEarthNet Spatial Split Isolation** | Spatially separated patch split ($\ge 12.0\text{ km}$ buffer in UTM Zone 34N); Tile `T34TCS` ($N=627$) excluded | `artifacts/benchmarks/spatial_buffer_audit.json`<br>`data/test_full_metadata.parquet` | `tests/test_m31_spatial_buffer.py` | `ce71342` | **AUDITED & VERIFIED** |
| **C7** | **OSCD Training Patch Count** | **586 regular grid patches** ($256 \times 256$, stride 64) from 10 training scenes with dynamic augmentation | `satquery/specialists/dataset_change_detection.py`<br>`artifacts/specialists/oscd_frozen_split_manifest.json` | `scripts/verify_oscd_splits.py` | `065575c` | **AUDITED & VERIFIED** |
| **C8** | **M3.2 Change-Detection FC-Siam-diff Macro F1** | **37.66%** ($+10.35\%$ over Gaussian CVA) | `artifacts/specialists/m32_change_detection_comparison.json`<br>`artifacts/specialists/m32_per_scene_evaluation.csv` | `tests/test_m32_change_detection_evaluation.py` | `065575c` | **AUDITED & VERIFIED** |
| **C9** | **M3.2 Change-Detection FC-Siam-diff Pooled F1** | **38.18%** (Precision: 35.33%, Recall: 41.53%, IoU: 23.60%) | `artifacts/specialists/m32_change_detection_comparison.json` | `scripts/reproduce_m32_evaluation_report.py` | `065575c` | **AUDITED & VERIFIED** |
| **C10** | **M3.2 Certification Target Status** | **CERTIFICATION TARGET NOT ACHIEVED** ($37.66\% < 50.00\%$) | `PHASE3_M32_SPECIALIST_EVALUATION_REPORT.md`<br>`artifacts/specialists/m32_change_detection_comparison.json` | `tests/test_m32_change_detection_evaluation.py` | `065575c` | **AUDITED & VERIFIED** |
| **C11** | **M3.2 BIT Transformer Underperformance** | **12.95%** Macro $F_1$, **12.60%** Pooled $F_1$ | `artifacts/specialists/m32_change_detection_comparison.json`<br>`artifacts/specialists/m32_per_scene_evaluation.csv` | `tests/test_m32_change_detection_evaluation.py` | `065575c` | **AUDITED & VERIFIED** |
| **C12** | **Physical Gate Incompatible Dispatch Rate** | $\text{IDER}_{\text{obs}} = \mathbf{0.0\%}$ ($0/28$ dispatches observed; 95% Wilson bound $8.81\%$, Binomial $10.15\%$) | `artifacts/benchmarks/n200_routing_baselines_summary.json`<br>`PHASE3_M30_PROTOCOL_FREEZE.md` | `tests/test_affine_full_footprint_gate.py` | `v0.2.0-phase2-freeze` | **AUDITED & VERIFIED** |
| **C13** | **Ambiguity Ground Truth Inter-Annotator Agreement** | Raw agreement $P_o = 97.4\%$, Cohen's $\kappa = \mathbf{0.918}$ | `data/benchmark_ambiguity_annotations.json` | `tests/test_ambiguity_annotations.py` | `ce71342` | **AUDITED & VERIFIED** |

---

## 2. Integrity Certification
All empirical claims listed above correspond directly to binary array outputs, confusion matrices, JSON manifests, and unit tests preserved in the repository. No claim is extrapolated beyond its verified test cohort or held-out spatial partition.
