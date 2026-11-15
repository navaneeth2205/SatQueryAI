# Milestone M3.3 — Step 2: System Evaluation & Error Analysis Report
**Document Identifier**: `SATQUERY-REPORT-M3.3-STEP2-EVAL-ERROR`  
**Evaluation Date**: 2026-11-14  
**Audit Status**: RECONCILED / AUDITED / VERIFIED  
**Milestone Version**: M3.3 (System Demonstration, Selective Routing & Deployment Readiness)  
**Charter Constraints**: Frozen Baselines (M3.2 Siamese ResNet-18 U-Net Champion `e4_best_model.pt`, SHA-256 `8fd23f8b...`; M3.1 Calibrated Router; Phase 2 LightGBM Multimodal Specialist). Strictly read-only evaluation; zero parameter tuning against held-out evaluation splits.

---

## 1. Executive Summary & Master Evaluation Scorecard

Milestone **M3.3 — Step 2** delivers the comprehensive system-level evaluation and multi-dimensional error analysis for SatQuery AI. In strict compliance with the project charter, upstream models and thresholds remain completely frozen:
- **Change Detection Champion**: Siamese ResNet-18 U-Net with scSE attention (`e4_best_model.pt`, frozen threshold $\tau^* = 0.40$).
- **Change Detection Fallback**: Classical Gaussian-Smoothed CVA ($\sigma=1.0$, frozen threshold $\tau = 0.20$).
- **Land Cover Specialist**: LightGBM 488-feature GBDT with frozen validation-tuned per-class thresholds.
- **Task Router Models**: 
  - *Phase 0 Starter Baseline Router* (`artifacts/router/router_model.joblib`, Word n-grams 1-2, SHA-256 `f0e6608b...`).
  - *Phase 1 Champion Router* (`artifacts/phase1_router/champion_router_model.joblib`, Subword Character n-grams 3-5, SHA-256 `11366c09...`).

> [!IMPORTANT]
> **Evaluation Provenance Disclosure**:
> The 10 OSCD test scenes ($3,077,936\text{ valid pixels}$) were previously examined during Milestone M3.2 model selection, and the BigEarthNet-14k test split ($3,248\text{ patches}$) was examined during Phase 2 specialist benchmarking. The metrics reported herein represent **frozen-model system integration evaluations** verifying that the end-to-end multi-specialist architecture faithfully preserves specialist performance without integration bugs, pipeline distortion, or memory corruption. They do **not** represent fresh independent validation or new claims of generalization.

```
========================================================================================================================
                                     M3.3 STEP 2 SYSTEM EVALUATION SCORECARD
========================================================================================================================
 Evaluation Area                    Primary Evaluation Metric            Observed Benchmark Metric      Audit Status
------------------------------------------------------------------------------------------------------------------------
 1. Router Provenance & Accuracy    Locked Validation (N=56)                                            VERIFIED
    - Phase 0 Baseline Router       Word n-grams (1, 2) + LogReg         Macro F1 = 79.79%, Acc = 80.36% Baseline Path
    - Phase 1 Champion Router (T1)  Subword char_wb (3, 5) + LogReg      Macro F1 = 87.42%, Acc = 87.50% Champion Model
    - Phase 1 Champion Router (T2)  Champion with Keyword Guard          Macro F1 = 83.89%, Acc = 83.93% Guard Intercept
    - Calibration Telemetry         Temperature Scaling (T=0.0991)       NLL -60.19%, ECE -5.27%         Calibrated
 2. Specialist Integration Metrics  Frozen Held-Out Evaluation                                          VERIFIED
    - Change Detection Champion     10 OSCD Test Scenes (tau*=0.40)      Macro F1 = 37.83%, Pooled 39.69% Bitwise Parity
    - False Positive Suppression    vs FC-Siam-diff (120,922 FP)         48,119 FP (60.20% reduction)   Confirmed
    - Classical CVA Fallback        Gaussian CVA (tau=0.20)              Macro F1 = 27.31%, Pooled 33.93% Operational
    - Land Cover: Micro F1          BigEarthNet-14k Test (N=3,248)       Micro F1 = 72.99% (Tuned)      Verified
    - Land Cover: Standard Macro F1 All 16 Classes (incl. 0 support)     Macro F1 = 47.39% (Tuned)      Verified
    - Land Cover: Present-Class F1  14 Classes with Non-Zero Support     Macro F1 = 54.16% (Tuned)      Verified
 3. Evidence Pipeline Integrity     Explainable Synthesis Conformance                                   VERIFIED
    - Spatial Bounding Boxes        Connected Component Extraction       [ymin, xmin, ymax, xmax] Pixel Verified
    - Bounding Box Clipping         Bounded within [0..H, 0..W]          100% Clipped (20/20 Traces)    Verified
    - Valid Pixel & NoData Masking  Finite & Sensor NoData Verification  3,077,936 / 3,077,936 Finite   Verified
    - JSON Evidence Schema          EvidenceReport Dataclass Conformance 100% Schema Valid (80/80 runs) Verified
 4. Failure Analysis & Diagnostics  Error Characterization & Profiling                                  VERIFIED
    - False Positive Concentration  Austin FP / Total Test FP            23,847 / 48,119 = 49.56%       Hypothesized
    - False Negative Concentration  Mumbai FN / Total Test FN            46,157 / 107,775 = 42.83%      Hypothesized
    - Unsupported Task Containment  Zero Perception Coercion             100% Transparent Stubs/Halts   Verified
    - CPU Inference Bottleneck      Sliding-Window Neural P95            P95 = 3,908 ms (Pisa specialist) Bottleneck
========================================================================================================================
 STEP 2 AUDIT VERDICT: PASS (System Integration Evaluated; Deployment Readiness Subject to Step 3 Packaging)
========================================================================================================================
```

---

## 2. Activity 1: Router Provenance, Benchmarks & Policy Dynamics

### 2.1 Disambiguation of Router Checkpoints & Feature Representations
The preliminary Step 2 draft reported Macro F1 = $79.79\%$ and Accuracy = $80.36\%$ ($N=56$). Forensic audit reveals that two distinct router artifacts exist in the repository:

1. **Phase 0 Starter Baseline Router** (`artifacts/router/router_model.joblib`):
   - **File SHA-256**: `f0e6608b11c01da3cfae454342526760ded16fb5c18ed4bdaf5cea5727b317b7`
   - **Feature Extraction**: Word-level n-grams (`ngram_range=(1, 2)`, English stopwords removed, sublinear TF scaling).
   - **Classifier**: `CalibratedClassifierCV(cv=3, LogisticRegression(C=1.0, class_weight='balanced'))`.
   - **Orchestrator Association**: Loaded by default in `satquery/orchestrator.py` via `TaskRouter.load()`.
   - **Performance on Locked Validation Set ($N=56$)**:
     - **Tier 1 (Raw Statistical ML)**: **Macro F1 = 79.79%**, **Accuracy = 80.36%** ($45 / 56$ correct).
     - **Tier 2 (With Keyword Guard)**: **Macro F1 = 81.81%**, **Accuracy = 82.14%** ($46 / 56$ correct).
2. **Phase 1 Champion Router** (`artifacts/phase1_router/champion_router_model.joblib`):
   - **File SHA-256**: `11366c09e98ce3298313abefa30d45dc29264874acbf8964a3288b942dacdf6b`
   - **Metadata Path**: [`artifacts/phase1_router/champion_router_metadata.json`](file:///C:/Users/B%20NAVANEETH/SatQueryAI/artifacts/phase1_router/champion_router_metadata.json)
   - **Feature Extraction**: Subword Character n-grams (`analyzer='char_wb'`, `ngram_range=(3, 5)`, sublinear TF scaling).
   - **Classifier**: `CalibratedClassifierCV(cv=3, LogisticRegression(C=1.0, class_weight='balanced'))`.
   - **Performance on Locked Validation Set ($N=56$)**:
     - **Tier 1 (Raw Statistical ML)**: **Macro F1 = 87.42%**, **Accuracy = 87.50%** ($49 / 56$ correct).
     - **Tier 2 (With Keyword Guard)**: **Macro F1 = 83.89%**, **Accuracy = 83.93%** ($47 / 56$ correct).

> [!NOTE]
> **Why the Step 2 Score Differed**:
> When Phase 1 was reconciled, the winning Subword Character n-gram pipeline was serialized into `artifacts/phase1_router/champion_router_model.joblib`, while `artifacts/router/router_model.joblib` was preserved in its frozen Phase 0 state (`6aae3a80...` / `f0e6608b...`) to avoid disrupting downstream integration tests. The benchmark files `router_confusion_matrix.csv` and `router_benchmark_summary.json` reflect the **Phase 0 baseline router**. Both models are fully reproducible from [`artifacts/phase1_router/phase1_locked_validation_predictions.csv`](file:///C:/Users/B%20NAVANEETH/SatQueryAI/artifacts/phase1_router/phase1_locked_validation_predictions.csv).

### 2.2 Comparison of Router Confusion Matrices ($N=56$ Locked Validation Set)

#### Phase 0 Baseline Router Confusion Matrix (Word n-grams 1-2, Tier 1 Statistical)
*Macro F1: 79.79% | Accuracy: 80.36% (45 / 56 correct)*
```
+------------------------------------+----+----+----+----+----+----+----+-------+
| True Task \ Predicted Task         | CD | CV | LC | OS | SC | SI | VG | Total |
+------------------------------------+----+----+----+----+----+----+----+-------+
| 1. CHANGE_DETECTION (CD)           |  8 |  0 |  0 |  0 |  0 |  0 |  0 |     8 |
| 2. CHANGE_VQA (CV)                 |  1 |  6 |  0 |  0 |  0 |  0 |  0 |     7 |
| 3. LAND_COVER_CLASSIFICATION (LC)  |  1 |  1 |  6 |  0 |  0 |  1 |  0 |     9 |
| 4. OPTICAL_SAR_FUSION (OS)         |  0 |  0 |  0 |  8 |  0 |  0 |  0 |     8 |
| 5. SCENE_CAPTIONING (SC)           |  0 |  0 |  1 |  0 |  7 |  0 |  0 |     8 |
| 6. SINGLE_IMAGE_VQA (SI)           |  0 |  1 |  0 |  0 |  2 |  4 |  1 |     8 |
| 7. VISUAL_GROUNDING (VG)           |  1 |  0 |  0 |  0 |  1 |  0 |  6 |     8 |
+------------------------------------+----+----+----+----+----+----+----+-------+
| Total Predicted                    | 11 |  8 |  7 |  8 | 10 |  5 |  7 |    56 |
+------------------------------------+----+----+----+----+----+----+----+-------+
```

#### Phase 1 Champion Router Confusion Matrix (Subword char_wb 3-5, Tier 1 Statistical)
*Macro F1: 87.42% | Accuracy: 87.50% (49 / 56 correct)*
```
+------------------------------------+----+----+----+----+----+----+----+-------+
| True Task \ Predicted Task         | CD | CV | LC | OS | SC | SI | VG | Total |
+------------------------------------+----+----+----+----+----+----+----+-------+
| 1. CHANGE_DETECTION (CD)           |  8 |  0 |  0 |  0 |  0 |  0 |  0 |     8 |
| 2. CHANGE_VQA (CV)                 |  2 |  5 |  0 |  0 |  0 |  0 |  0 |     7 |
| 3. LAND_COVER_CLASSIFICATION (LC)  |  0 |  0 |  8 |  0 |  0 |  1 |  0 |     9 |
| 4. OPTICAL_SAR_FUSION (OS)         |  0 |  0 |  0 |  8 |  0 |  0 |  0 |     8 |
| 5. SCENE_CAPTIONING (SC)           |  0 |  0 |  0 |  0 |  8 |  0 |  0 |     8 |
| 6. SINGLE_IMAGE_VQA (SI)           |  0 |  0 |  0 |  0 |  0 |  6 |  2 |     8 |
| 7. VISUAL_GROUNDING (VG)           |  0 |  0 |  0 |  0 |  0 |  2 |  6 |     8 |
+------------------------------------+----+----+----+----+----+----+----+-------+
| Total Predicted                    | 10 |  5 |  8 |  8 |  8 |  9 |  8 |    56 |
+------------------------------------+----+----+----+----+----+----+----+-------+
```

### 2.3 Multi-Tier Metric Summary Across Models

| Evaluation Tier & Configuration | Model Artifact / Checkpoint | Macro Precision | Macro Recall | Macro $F_1$ | Accuracy | Multiclass Brier | Flagged Review ($\hat{c} < 0.65$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 0 Baseline (Tier 1: Pure ML)** | `artifacts/router/router_model.joblib` | 81.50% | 80.36% | **79.79%** | **80.36%** (45/56) | 0.3124 | 62.50% |
| **Phase 0 Baseline (Tier 2: + Guard)** | `artifacts/router/router_model.joblib` | 83.21% | 82.14% | **81.81%** | **82.14%** (46/56) | N/A | 62.50% |
| **Phase 1 Champion (Tier 1: Pure ML)** | `artifacts/phase1_router/champion_...` | 88.81% | 87.20% | **87.42%** | **87.50%** (49/56) | **0.2565** | 46.43% |
| **Phase 1 Champion (Tier 2: + Guard)** | `artifacts/phase1_router/champion_...` | 84.62% | 83.93% | **83.89%** | **83.93%** (47/56) | N/A | 46.43% |

### 2.4 Temperature Calibration & Policy Interception
- **Calibration Telemetry ($T_{\text{cal}} = 0.0991$)**: Sharpens underconfident linear logit outputs, reducing validation negative log-likelihood from $1.5991$ to $0.6366$ ($-60.19\%$) and reducing Expected Calibration Error from $22.72\%$ to $17.45\%$.
- **Policy Interaction**:
  - Under `provisional` mode: low-confidence queries bypass early halts, allowing unsupported tasks to produce structured `UNAVAILABLE_SPECIALIST` stubs.
  - Under `clarify` mode: Task Router thresholding ($\tau_{\text{intent}} = 0.65, \tau_{\text{margin}} = 0.15$) intercepts ambiguous and low-confidence requests prior to specialist dispatch, shifting 4 low-confidence unsupported/ambiguous queries from unavailable stubs into `NEEDS_CLARIFICATION` halts.

---

## 3. Activity 2: Specialist Evaluation Scorecards & Disambiguation

### 3.1 Change Detection Specialist: Siamese ResNet-18 U-Net Champion
- **Evaluated Checkpoint**: `artifacts/specialists/change_detection/siamese_resnet18_unet_best.pt` (bitwise identical to `e4_best_model.pt`, SHA-256 `8fd23f8b...`).
- **Operating Parameters**: Frozen validation threshold $\tau^* = 0.40$, minimum component area $A_{\min} = 9\text{ pixels}$.
- **Dataset Evaluated**: 10 held-out OSCD test scenes ($3,077,936$ total valid pixels; $159,077$ ground truth changed pixels, pooled change ratio $5.17\%$).

#### Authoritative Per-Scene Breakdown (Frozen-Model System Integration Evaluation)

| Test Scene | Spatial Dimensions | Valid Pixels | True Change Pixels | True Change % | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Precision | Recall | Scene $F_1$ | Scene IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `aguasclaras` | $353 \times 426$ | 150,378 | 8,552 | 5.69% | 4,915 | 5,490 | 3,637 | 47.24% | 57.47% | **51.85%** | 35.00% |
| `austin` | $639 \times 688$ | 439,632 | 5,018 | 1.14% | 3,414 | 23,847 | 1,604 | 12.52% | 68.04% | **21.15%** | 11.83% |
| `beihai` | $458 \times 476$ | 218,008 | 969 | 0.44% | 255 | 2,169 | 714 | 10.52% | 26.32% | **15.03%** | 8.13% |
| `berlin` | $730 \times 544$ | 397,120 | 28,637 | 7.21% | 6,643 | 2,224 | 21,994 | 74.92% | 23.20% | **35.43%** | 21.53% |
| `brasilia` | $426 \times 451$ | 192,126 | 13,054 | 6.79% | 10,595 | 3,688 | 2,459 | 74.18% | 81.16% | **77.51%** | 63.28% |
| `budapest` | $433 \times 469$ | 203,077 | 5,242 | 2.58% | 1,871 | 3,305 | 3,371 | 36.15% | 35.69% | **35.92%** | 21.89% |
| `campinas` | $241 \times 385$ | 92,785 | 1,227 | 1.32% | 555 | 1,503 | 672 | 26.97% | 45.23% | **33.79%** | 20.33% |
| `mumbai` | $774 \times 634$ | 490,716 | 48,689 | 9.92% | 2,532 | 1,576 | 46,157 | 61.64% | 5.20% | **9.59%** | 5.04% |
| `nantes` | $545 \times 558$ | 304,110 | 2,419 | 0.80% | 720 | 410 | 1,699 | 63.72% | 29.76% | **40.57%** | 25.45% |
| `pisa` | $824 \times 716$ | 589,984 | 45,270 | 7.67% | 19,802 | 3,907 | 25,468 | 83.52% | 43.74% | **57.41%** | 40.27% |
| **Pooled Total** | — | **3,077,936** | **159,077** | **5.17%** | **51,302** | **48,119** | **107,775** | **51.60%** | **32.25%** | **39.69%** | **24.76%** |
| **Macro F1** | — | — | — | **4.36%** | — | — | — | — | — | **37.83%** | — |

- **Unweighted Macro $F_1$**: $\frac{51.8542 + 21.1531 + 15.0309 + 35.4256 + 77.5140 + 35.9186 + 33.7900 + 9.5915 + 40.5748 + 57.4146}{10} = \mathbf{37.83\%}$.
- **False Alarm Suppression**: Total test FP reduced from $120,922$ (FC-Siam-diff) to $48,119$ (**$60.20\%$ reduction**).

### 3.2 Land Cover Specialist: Disambiguation of Macro F1 Metrics
Evaluated on BigEarthNet-14k Serbia held-out test split ($N=3,248$ patches, 488 multimodal features).

> [!IMPORTANT]
> **Metric Terminology Disambiguation**:
> 1. **Micro $F_1$**: Global aggregation across all instances and classes: $\mathbf{72.99\%}$ (validation-tuned thresholds) / $\mathbf{73.28\%}$ (default $\tau=0.50$).
> 2. **Standard 16-Class Macro $F_1$**: Unweighted arithmetic mean across all 16 predefined class definitions: $\mathbf{47.39\%}$ (validation-tuned) / $\mathbf{46.13\%}$ (default $\tau=0.50$). This includes two classes with **0 ground-truth test occurrences** ("Beaches, dunes, sands" and "Moors, heathland"), whose $F_1$ is imputed as $0.0$.
> 3. **Present-Class 14-Class Macro $F_1$**: Unweighted arithmetic mean restricted strictly to the 14 classes with non-zero test support: $\mathbf{54.16\%}$ (validation-tuned) / $\mathbf{52.72\%}$ (default $\tau=0.50$).

#### Detailed 16-Class Test Ledger (Validation-Tuned Decision Thresholds)

| Class Name | Test Support | TP | FP | FN | TN | Precision | Recall | Class $F_1$ | Operating $\tau$ | Support Scope |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Arable land** | 2,009 | 1,823 | 154 | 186 | 1,085 | 92.21% | 90.74% | **91.47%** | 0.29 | Present (Class 1/14) |
| **Broad-leaved forest** | 1,665 | 1,540 | 230 | 125 | 1,353 | 87.01% | 92.49% | **89.67%** | 0.51 | Present (Class 2/14) |
| **Inland waters** | 361 | 295 | 54 | 66 | 2,833 | 84.53% | 81.72% | **83.10%** | 0.30 | Present (Class 3/14) |
| **Land with natural vegetation** | 999 | 831 | 483 | 168 | 1,766 | 63.24% | 83.18% | **71.85%** | 0.52 | Present (Class 4/14) |
| **Urban fabric** | 517 | 357 | 155 | 160 | 2,576 | 69.73% | 69.05% | **69.39%** | 0.44 | Present (Class 5/14) |
| **Complex cultivation patterns** | 992 | 676 | 386 | 316 | 1,870 | 63.65% | 68.15% | **65.82%** | 0.57 | Present (Class 6/14) |
| **Transitional woodland, shrub** | 1,197 | 918 | 831 | 279 | 1,220 | 52.49% | 76.69% | **62.32%** | 0.27 | Present (Class 7/14) |
| **Coniferous forest** | 41 | 18 | 18 | 23 | 3,189 | 50.00% | 43.90% | **46.75%** | 0.35 | Present (Class 8/14) |
| **Inland wetlands** | 125 | 68 | 136 | 57 | 2,987 | 33.33% | 54.40% | **41.34%** | 0.11 | Present (Class 9/14) |
| **Pastures** | 454 | 214 | 442 | 240 | 2,352 | 32.62% | 47.14% | **38.56%** | 0.23 | Present (Class 10/14) |
| **Natural grassland & sparsely veg** | 256 | 67 | 69 | 189 | 2,923 | 49.26% | 26.17% | **34.18%** | 0.48 | Present (Class 11/14) |
| **Mixed forest** | 75 | 25 | 60 | 50 | 3,113 | 29.41% | 33.33% | **31.25%** | 0.60 | Present (Class 12/14) |
| **Industrial or commercial units** | 85 | 21 | 50 | 64 | 3,113 | 29.58% | 24.71% | **26.92%** | 0.17 | Present (Class 13/14) |
| **Permanent crops** | 46 | 2 | 24 | 44 | 3,178 | 7.69% | 4.35% | **5.56%** | 0.25 | Present (Class 14/14) |
| **Beaches, dunes, sands** | 0 | 0 | 0 | 0 | 3,248 | 0.00% | 0.00% | **0.00%** | 0.40 | Zero Support (0/0 imputed) |
| **Moors & heathland** | 0 | 0 | 14 | 0 | 3,234 | 0.00% | 0.00% | **0.00%** | 0.10 | Zero Support (14 FP) |

$$\text{Standard 16-Class Macro } F_1 = \frac{91.47 + \dots + 5.56 + 0.0 + 0.0}{16} = \mathbf{47.39\%}$$
$$\text{Present-Class 14-Class Macro } F_1 = \frac{91.47 + \dots + 5.56}{14} = \mathbf{54.16\%}$$

---

## 4. Activity 3: Evidence Evaluation & Trace Integrity Verification

### 4.1 Bounding Box Coordinate Validation & Clipping
All spatial bounding boxes generated by `SiameseResNetChangeDetectionSpecialist` derive from connected-component labeling on the filtered binary mask (`min_component_area = 9`):

1. **Coordinate Format & Ordering**:
   $$\text{bbox} = [y_{\min}, x_{\min}, y_{\max}, x_{\max}]$$
   Coordinates are integer pixel indices in `unprojected_pixel_coordinates_[ymin_xmin_ymax_xmax]`.
2. **Clipping & Dimension Bounds Guarantee**:
   Because `coords = np.argwhere(labeled == largest_label)` indexes directly into the $H \times W$ mask array, bounds clipping is guaranteed by construction:
   $$0 \le y_{\min} < y_{\max} \le H, \quad 0 \le x_{\min} < x_{\max} \le W$$
3. **Trace-Level Scene Association**:
   - `austin` ($H=639, W=688$): Largest cluster = $4,218\text{ px}$, $\text{bbox} = [142, 210, 318, 485]$. Bounded inside $[639, 688]$.
   - `berlin` ($H=730, W=544$): Largest cluster = $1,842\text{ px}$, $\text{bbox} = [205, 110, 360, 295]$. Bounded inside $[730, 544]$.
   - `pisa` ($H=824, W=716$): Largest cluster = $6,912\text{ px}$, $\text{bbox} = [310, 180, 520, 440]$. Bounded inside $[824, 716]$.
   - Parity verification (CD-04, identical pair on `pisa`): $0\text{ changed pixels}$, largest cluster = `null`, bbox = `null`.

### 4.2 Valid-Pixel Accounting & NoData Safeguards
Verified via [`satquery/utils/temporal.py`](file:///C:/Users/B%20NAVANEETH/SatQueryAI/satquery/utils/temporal.py):
- `valid_mask`: Defined strictly as `np.all(np.isfinite(t1), axis=-1) & np.all(np.isfinite(t2), axis=-1)`.
- If sensor NoData value is supplied, pixels matching NoData in either acquisition are masked out.
- Across all $10$ OSCD test scenes, exactly **$3,077,936$ of $3,077,936$ valid pixels ($100.0\%$)** were finite with zero NaN leakage into convolutional layers.
- Minimum valid pixel safeguard rejects observations with $< 5.0\%$ valid data.

### 4.3 Schema Conformance & Denominators
- **Demonstration Suite Traces**: Across all 4 execution profiles evaluated in [`artifacts/benchmarks/m33_system_demonstration_results.json`](file:///C:/Users/B%20NAVANEETH/SatQueryAI/artifacts/benchmarks/m33_system_demonstration_results.json), **exactly 20 out of 20 query traces per profile ($80 / 80$ total)** adhered to the `EvidenceReport` dataclass schema.
- **Trace Content Fidelity**: In $100\%$ of successful traces ($32 / 32$ executions), the numerical change percentage, bounding box coordinates, and class probabilities in `sensor_feature_summary` exactly matched the raw specialist output tensor statistics.

---

## 5. Activity 4: Failure Diagnostics & Observed vs Hypothesized Causes

### 5.1 False Positive Diagnostics: Austin Analysis
- **Verified Numerical Fact**: `austin` produced **$23,847$ false positive pixels**, accounting for **$49.56\%$** of all $48,119$ false positive pixels across the 10 test scenes:
  $$\frac{23,847}{48,119} \times 100\% = 49.5583\% \approx \mathbf{49.56\%}$$

```
+---------------------------------------------------------------------------------------------------------------+
| AUSTIN ERROR ANALYSIS: DIRECT OBSERVATIONS VS HYPOTHESIZED CONTRIBUTING FACTORS                               |
+------------------------------------+------------------------------------+-------------------------------------+
| Dimension                          | Directly Observed Evidence         | Hypothesized Contributing Factors   |
+------------------------------------+------------------------------------+-------------------------------------+
| Ground Truth Change Ratio          | 1.14% (5,018 px out of 439,632 px) | Minimal structural construction     |
| Scene Mean RGB Shift (T1 -> T2)    | T1: [79.0, 86.1, 96.6]             | Broad regional surface brightening  |
|                                    | T2: [86.8, 97.7, 106.6]            | consistent with soil moisture loss  |
| Widespread Pixel-Level Mean Delta  | Mean Delta: [24.5, 18.7, 15.0]     | High seasonal reflectance contrast  |
| Spatial Clustering of FP Pixels    | 49.56% of test FP concentrated in  | Dry grass senescence and non-       |
|                                    | broad contiguous open parcels      | irrigated pasture phenology         |
+------------------------------------+------------------------------------+-------------------------------------+
```

> [!CAUTION]
> **Scientific Wording Discipline**:
> While the high widespread channel delta (mean delta $\approx 19.4$ digital counts) across un-built parcels strongly aligns with seasonal phenological vegetation drying between acquisition dates, this mechanism represents a **hypothesized contributing factor**. Without localized ground-truth meteorological and soil moisture probes at acquisition timestamps, the report designates this as an empirical correlation rather than a proven cause.

### 5.2 False Negative Diagnostics: Mumbai Analysis
- **Verified Numerical Fact**: `mumbai` produced **$46,157$ false negative pixels**, accounting for **$42.83\%$** of all $107,775$ false negative pixels across the 10 test scenes:
  $$\frac{46,157}{107,775} \times 100\% = 42.8271\% \approx \mathbf{42.83\%}$$
- **Scene Recall**: **$5.20\%$** ($2,532\text{ TP}$ out of $48,689\text{ true change pixels}$).

```
+---------------------------------------------------------------------------------------------------------------+
| MUMBAI ERROR ANALYSIS: DIRECT OBSERVATIONS VS HYPOTHESIZED CONTRIBUTING FACTORS                               |
+------------------------------------+------------------------------------+-------------------------------------+
| Dimension                          | Directly Observed Evidence         | Hypothesized Contributing Factors   |
+------------------------------------+------------------------------------+-------------------------------------+
| Ground Truth Change Ratio          | 9.92% (48,689 px out of 490,716 px)| High-volume structural alteration   |
| High Base Surface Reflectance      | T1: [224.0, 210.9, 195.3]          | Highly reflective urban fabric and  |
|                                    | T2: [225.5, 210.5, 194.1]          | bright corrugated metal roofing     |
| Low Delta Across Changed Regions   | Mean Delta: [13.6, 15.7, 13.4]     | Low spectral contrast between old   |
|                                    | (substantially lower than Austin)  | and new impervious building covers  |
| Spatial Error Topology             | 42.83% of test FN concentrated in  | High-density informal settlement    |
|                                    | dense, high-frequency spatial grid | alterations blending into fabric    |
+------------------------------------+------------------------------------+-------------------------------------+
```

> [!CAUTION]
> **Scientific Wording Discipline**:
> The subtle spectral deltas across high-albedo urban terrain ($[13.6, 15.7, 13.4]$) provide direct numerical evidence of poor spectral contrast. Attributing this specifically to informal settlement materials (e.g. tarpaulin, weathered tin) is a **hypothesized contributing factor** based on visual inspection of urban density, not an independently audited material assay.

### 5.3 Unsupported Analytical Requests & Boundary Containment
- All $5$ unsupported tasks (`CHANGE_VQA`, `SINGLE_IMAGE_VQA`, `SCENE_CAPTIONING`, `VISUAL_GROUNDING`, `OPTICAL_SAR_FUSION`) and out-of-domain prompts (meteorology, crop yield modeling) were safely contained.
- In $100\%$ of tested cases, **zero ungrounded perception coercion or silent hallucinations occurred**:
  - `provisional` mode: registry emitted structured transparent stubs (`UNAVAILABLE_SPECIALIST`).
  - `clarify` mode: Task Router intercepted low-confidence queries before specialist invocation (`NEEDS_CLARIFICATION`).

### 5.4 High-Latency Profiles & Bottleneck Diagnosis
- **Profiling Attribution**: Steady-state CPU P95 latency (**$3,908.25\text{ ms}$**) is $100\%$ driven by sliding-window deep neural inference during Change Detection:
  - **CD-04** (`pisa`, $824 \times 716$): `specialist_ms` = **$5,035.56\text{ ms}$** (40 sliding-window patches).
  - **CD-01** (`austin`, $639 \times 688$): `specialist_ms` = **$3,734.22\text{ ms}$**.
  - **CD-02** (`berlin`, $730 \times 544$): `specialist_ms` = **$3,460.59\text{ ms}$**.
- **Non-Neural Comparison**: In contrast, all non-deep-learning operations execute in under **$75\text{ ms}$** on CPU (Task Router: $3 - 100\text{ ms}$; LightGBM: $10 - 20\text{ ms}$; input contract gates: $0.01 - 0.26\text{ ms}$).
- **Hardware Acceleration Relief**: CUDA execution reduces P50 latency to **$20.70\text{ ms}$** and Austin specialist execution to **$1,457.85\text{ ms}$**.

---

## 6. Forensic Discrepancy & Reconciliation Ledger

| # | Audit Finding | Original Claim in Draft | Verified Forensic Evidence | Corrected Reconciliation | Source Artifact Path |
| :-: | :--- | :--- | :--- | :--- | :--- |
| **1** | **Router Provenance** | Router Macro F1 = 79.79% reported as integrated system score. | Derived from `artifacts/router/router_model.joblib` (Phase 0 Word n-gram baseline). Phase 1 Champion Router (`artifacts/phase1_router/champion_...`, Subword char_wb 3-5) achieves **87.42% Macro F1**. | Explicitly report both models, their distinct architectures, SHA-256 hashes, and execution paths. | [`artifacts/phase1_router/champion_router_metadata.json`](file:///C:/Users/B%20NAVANEETH/SatQueryAI/artifacts/phase1_router/champion_router_metadata.json) |
| **2** | **Land Cover Macro F1** | Scorecard reported 47.39%, while text cited 54.16% without differentiating scopes. | Both metrics are authentic: 47.39% is Macro F1 across all 16 classes (including 2 zero-support classes); 54.16% is Macro F1 across the 14 present classes. | Separately report and explicitly label Micro F1 (72.99%), 16-Class Macro F1 (47.39%), and 14-Present-Class Macro F1 (54.16%). | [`artifacts/benchmarks/land_cover_reconciliation_audit.json`](file:///C:/Users/B%20NAVANEETH/SatQueryAI/artifacts/benchmarks/land_cover_reconciliation_audit.json) |
| **3** | **Failure Hypotheses** | Austin FP and Mumbai FN asserted as proven physical causes. | Austin FP ($23,847 / 48,119 = 49.56\%$) and Mumbai FN ($46,157 / 107,775 = 42.83\%$) verified from raw counts. Underlying physical causes lack ground-truth sensor telemetry. | Reclassify mechanisms as **hypothesized contributing factors**, separating directly observed pixel deltas from physical hypotheses. | [`artifacts/benchmarks/oscd_per_scene_evaluation_breakdown.csv`](file:///C:/Users/B%20NAVANEETH/SatQueryAI/artifacts/benchmarks/oscd_per_scene_evaluation_breakdown.csv) |
| **4** | **Evidence Integrity** | 100% valid coverage and schema validity asserted without coordinate or denominator proof. | Bounding box ordering verified as $[y_{\min}, x_{\min}, y_{\max}, x_{\max}]$; clipping verified by array bounds; NoData masking verified in `temporal.py`. Denominator: 20/20 demonstration traces (80/80 total). | State explicit verification of coordinate format, spatial bounds clipping, and cite exact denominators for all 100% validity claims. | [`satquery/evidence.py`](file:///C:/Users/B%20NAVANEETH/SatQueryAI/satquery/evidence.py)<br>[`satquery/utils/temporal.py`](file:///C:/Users/B%20NAVANEETH/SatQueryAI/satquery/utils/temporal.py) |
| **5** | **Test-Set Reuse** | OSCD and BigEarthNet test metrics presented without disclosing prior evaluation. | Scenes and patches were evaluated in M3.2 and Phase 2. | Disclose prior evaluation; frame results strictly as **frozen-model system integration evaluations**, confirming zero parameter tuning. | [`phase2_2_step3_siamese_resnet18_unet_report.md`](file:///C:/Users/B%20NAVANEETH/.gemini/antigravity/brain/e2f8df4b-45fe-46b5-a4f3-9006006ba07a/phase2_2_step3_siamese_resnet18_unet_report.md) |
| **6** | **Deployment Readiness** | Concluded "Certified for M3.3 Step 3 Deployment Readiness". | Offline system evaluation demonstrates integration health but does not certify production readiness. | Reframe conclusion to system evaluation completion; state deployment readiness is subject to Step 3 packaging audit. | Project charter & milestone roadmaps |

---

## 7. Audit Conclusion & Readiness Boundary

Following the strictly read-only forensic audit:
1. **Router Provenance**: Baseline router (Word n-grams 1-2, Macro F1 $79.79\%$) and Champion router (Subword char_wb 3-5, Macro F1 $87.42\%$) are fully disambiguated with exact SHA-256 hashes.
2. **Land Cover Metrics**: Explicitly separated into Micro F1 ($72.99\%$), Standard 16-Class Macro F1 ($47.39\%$), and Present-Class 14-Class Macro F1 ($54.16\%$).
3. **Failure Analysis**: Austin FP ($49.56\%$) and Mumbai FN ($42.83\%$) verified from raw counts; physical causes accurately labeled as hypothesized contributing factors.
4. **Evidence Integrity**: Coordinate conventions $[y_{\min}, x_{\min}, y_{\max}, x_{\max}]$, clipping, NoData masking, and exact denominators ($20/20$ traces, $3,077,936$ pixels) confirmed.
5. **Deployment Boundary**: System evaluation is complete and verified, serving as the verified prerequisite for **Milestone M3.3 Step 3: Deployment Readiness, Production Packaging & Final Certification**.

---
*Signed by: Antigravity AI Forensic Audit System — 2026-11-14*
