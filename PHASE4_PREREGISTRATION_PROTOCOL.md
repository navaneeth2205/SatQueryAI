# SatQuery AI — Phase 4 Preregistration Protocol: Robustness and Generalization Validation

**Document Identifier**: `SATQUERY-P4-PROTOCOL-V1.0-DRAFT`  
**Phase**: Phase 4 — Robustness and Generalization Validation  
**Date**: November 8, 2026  
**Status**: **PROPOSED PREREGISTRATION PROTOCOL — DRAFT v1.0 (Awaiting User Freeze)**  
**Repository Branch**: `feature/phase4-generalization-protocol`  
**Preceding Milestones**: Phase 3 Closed (M3.1 ASL/CB-BCE, M3.2 FC-Siam-diff, M3.3 Selective Routing) with Pre-Registered Limitations  
**Model Weight Provenance**: Completely Frozen from Phase 3 (`artifacts/specialists/deep_land_cover_asl/best_model_asl.pt`, `artifacts/specialists/deep_change_detection_fc_siam_diff/best_model.pt`, `artifacts/router/router_model.joblib`)

---

## 1. Executive Research Motivation & Non-Tuning Directive

Phase 3 established the core decoupled perception and input-constrained routing architecture of SatQuery AI:
- **Milestone M3.1**: Demonstrated multi-label land-cover classification using asymmetric loss (ASL) and class-balanced BCE on Sentinel-2 patches.
- **Milestone M3.2**: Established deep Siamese change detection using FC-Siam-diff on bitemporal Sentinel-2 imagery.
- **Milestone M3.3**: Established a temperature-calibrated linear router with a deterministic 8-dimensional physical compatibility gate, demonstrating $0.00\%$ observed incompatible dispatches ($\text{IDER}_{\text{obs}} = 0.0\%$, Wilson 95% bound $\le 9.89\%$) and $59.0\%$ 7-task architectural success.

### The Non-Tuning Scientific Directive
**Phase 4 is strictly a validation, stress-testing, and operational envelope discovery phase.**  
In accordance with open-science pre-registration principles:
1. **Zero Weight / Threshold Modification**: No model weights, router calibration temperatures ($T_{\text{cal}} = 0.0991$), or routing operating thresholds ($\tau_{\text{intent}}=0.65, \tau_{\text{margin}}=0.15, \tau_{\text{sel}}=1.0$) will be tuned or optimized against Phase 4 evaluation data.
2. **Confirmatory vs. Exploratory Segregation**: All primary hypotheses (H1–H5) are confirmatory evaluations of the frozen Phase 3 checkpoints. Any architectural modifications, fine-tuning, or prompt adaptations will be quarantined into separate post-hoc exploratory studies.
3. **Truthful Negative Reporting**: Performance degradations, domain shift collapses, or edge-case gate failures will be published transparently as characterizations of the system's operational envelope.

---

## 2. Core Decisions & Strategic Stance

### Decision 1: Evaluation Datasets & Regional Stratification
- **Track 4.1 (Geographic Land Cover)**: Use geographically disjoint Sentinel-2 Level-2A imagery across 4 distinct global biomes:
  - *Biome A (Arid / Desert)*: Middle East / North Africa (e.g., Cairo, Dubai outskirts, Sahara transition).
  - *Biome B (Tropical / Rainforest)*: Equatorial South America / Central Africa (e.g., Amazon basin, Congo basin).
  - *Biome C (Boreal / Sub-Arctic)*: Northern Scandinavia / Alaska (e.g., Kiruna, Fairbanks).
  - *Biome D (Monsoon / Humid Subtropical)*: South/Southeast Asia (e.g., Ganges delta, Mekong).
- **Track 4.1 (Geographic Change Detection)**: Evaluate bitemporal Sentinel-2 scenes from **S2Looking** and held-out global cities outside the European/Mediterranean concentration of OSCD, ensuring complete disjointness from OSCD training/val/test scenes.

### Decision 2: Sensor Scope — Two-Tier Confirmatory vs. Exploratory Design
- **Tier 1 (Confirmatory Safety Enforcement)**: Native Landsat-8/9 ($30\text{ m}$ multispectral) and PlanetScope ($3\text{ m}$ RGB/NIR) will be evaluated as **unsupported native inputs**. The confirmatory test checks whether the physical compatibility gate $\mathcal{C}(t, o)$ strictly rejects these observations (`ERR_RESOLUTION_INCOMPATIBLE`, `ERR_INSUFFICIENT_BANDS`), verifying zero incompatible dispatches ($\text{IDER} = 0.0\%$).
- **Tier 2 (Exploratory Domain Compatibility)**: An exploratory track evaluates Harmonized Landsat-Sentinel (HLS) Level-30 data resampled to $10\text{ m}$ to measure perception degradation without treating resampling as true physical equivalence.

### Decision 3: Compute Budget & Hardware Confinement
- All Phase 4 confirmatory tracks are **inference-only evaluations** of frozen models.
- The evaluation will execute entirely on the local development workstation equipped with an **NVIDIA GeForce RTX 2050 Laptop GPU (4,096 MiB VRAM)** and CUDA 12.8.
- Inference benchmarks:
  - ASL ResNet-50 inference: $\approx 15\text{ ms/batch}$ ($< 1.2\text{ GB}$ VRAM).
  - FC-Siam-diff inference: $\approx 20\text{ ms/batch}$ ($< 1.0\text{ GB}$ VRAM).
  - Task Router inference: $< 1\text{ ms/query}$ on CPU.
- Local execution ensures $100\%$ deterministic reproducibility without external cloud API dependencies.

### Decision 4: Research Ambition
- Primary goal: **Measure the true robustness boundaries and operational failure modes** of the existing frozen system.
- Secondary goal: Document architectural failure modes as actionable scientific findings for future work.

---

## 3. Detailed Experimental Tracks & Hypotheses

```mermaid
flowchart TD
    subgraph P4["Phase 4 Experimental Validation Tracks"]
        T1["Track 4.1: Cross-Region Geographic Transfer (H1)"]
        T2["Track 4.2: Cross-Sensor & Resolution Defense (H2)"]
        T3["Track 4.3: Atmospheric & Quality Stress Testing (H3)"]
        T4["Track 4.4: Complex & Adversarial Linguistic Query Defense (H4)"]
        T5["Track 4.5: Adversarial Physical Metadata Perturbations (H5)"]
    end

    T1 --> MET1["Perception Degradation Delta F1, Spatial Invariance"]
    T2 --> MET2["Safety Gate IDER = 0.0%, Resolution Rejection"]
    T3 --> MET3["Noise Curves: Cloud & Radiometric Robustness"]
    T4 --> MET4["Routing Clarification Precision/Recall, Zero Coercion"]
    T5 --> MET5["Physical Gate Robustness: Sub-pixel, CRS, NaN Checks"]
```

---

### Track 4.1: Cross-Region Geographic Generalization

- **Hypothesis (H1)**: While perception accuracy (Macro $F_1$, IoU) will degrade under severe biomic and seasonal distribution shift compared to in-domain validation, the physical compatibility gate and router will maintain invariant validity detection ($0\%$ false dispatches).
- **Target Population**:
  - Land Cover: $N = 2,000$ Sentinel-2 patches ($120 \times 120$ pixels, 4 bands: B02, B03, B04, B08) evenly stratified across 4 global biomes (500 per biome).
  - Change Detection: $N = 10$ held-out non-European bitemporal city scenes ($N \ge 500$ patches) from S2Looking / global Sentinel-2 acquisitions.
- **Evaluation Metrics**:
  - Land Cover: Macro $F_1$, Present Macro $F_1$, Hamming Loss, Per-class Precision/Recall across biomes.
  - Change Detection: Scene-wise Macro $F_1$, Pooled Precision, Recall, and IoU.
  - Transfer Delta: $\Delta F_1 = F_1^{(\text{out-of-domain})} - F_1^{(\text{in-domain})}$.
- **Pre-Registered Acceptance Criteria**:
  - Land Cover: System exhibits graceful degradation ($\text{Macro } F_1 \ge 25.0\%$ on arid and temperate transfer biomes).
  - Change Detection: Bitemporal alignment gate $\mathcal{C}(t, o)$ executes with $100\%$ precision on CRS and grid alignment.

---

### Track 4.2: Cross-Sensor & Cross-Resolution Compatibility

- **Hypothesis (H2)**: The physical compatibility gate $\mathcal{C}(t, o)$ will deterministically intercept $100\%$ of native Landsat-8/9 ($30\text{ m}$) and PlanetScope ($3\text{ m}$) observations targeting Sentinel-2 specialists, achieving $\text{IDER}_{\text{obs}} = 0.00\%$.
- **Target Population**:
  - $N = 100$ synthetic and real sensor observation payloads:
    - 40 Landsat-8/9 Level-1/2 payloads (GSD $30\text{ m}$, 11 spectral bands).
    - 40 PlanetScope Scene payloads (GSD $3\text{ m}$, 4 or 8 spectral bands).
    - 20 Sentinel-2 Level-2A control payloads (GSD $10\text{ m}$).
- **Evaluation Metrics**:
  - Incompatible Dispatch Error Rate ($\text{IDER}_{\text{obs}}$).
  - Constraint Violation Detection Recall ($\text{CVDR}_{\text{obs}}$).
  - Error code classification fidelity (`ERR_RESOLUTION_INCOMPATIBLE`, `ERR_INSUFFICIENT_BANDS`).
- **Pre-Registered Acceptance Criteria**:
  - Confirmatory safety: $\text{IDER}_{\text{obs}} = 0.00\%$ ($0$ incompatible dispatches across all 80 off-sensor payloads; finite-sample Wilson 95% bound $\le 4.5\%$).

---

### Track 4.3: Atmospheric & Image-Quality Stress Testing

- **Hypothesis (H3)**: As synthetic cloud contamination and atmospheric haze increase, specialist performance will decrease monotonically, but the cloud-fraction guard in $\mathcal{C}(t, o)$ will halt observations exceeding $20\%$ cloud cover for Land Cover and $10\%$ for Change Detection before execution.
- **Degradation Regimes**:
  - Regime A (Cloud Contamination): Cloud masks injected at fractions $f_c \in \{0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50\}$.
  - Regime B (Atmospheric Haze): Atmospheric scattering modeled via dark-channel haze injection ($I'(x) = I(x)t(x) + A(1 - t(x))$ for $t \in \{0.9, 0.7, 0.5, 0.3\}$).
  - Regime C (Sensor Noise): Gaussian radiometric noise $\sigma \in \{0.01, 0.02, 0.05, 0.10\}$ and salt-and-pepper band dropout.
- **Target Population**:
  - $N = 500$ clean control patches evaluated across all degradation steps ($500 \times 15 = 7,500$ evaluated instances).
- **Evaluation Metrics**:
  - Graceful Degradation Curve: $F_1(\text{degradation level})$.
  - Cloud Gate Interception Rate: Accuracy of `ERR_CORRUPT_PAYLOAD` rejection at the pre-registered thresholds.
- **Pre-Registered Acceptance Criteria**:
  - Cloud threshold compliance: $100\%$ rejection of payloads with cloud fraction $> 20\%$ for Land Cover and $> 10\%$ for Change Detection.

---

### Track 4.4: Complex & Adversarial Linguistic Query Defense

- **Hypothesis (H4)**: Calibrated confidence and margin thresholds ($\tau_{\text{intent}} = 0.65, \tau_{\text{margin}} = 0.15$) will intercept complex, polysemous, and adversarial prompt injections via interactive clarification (`CLARIFY_AMBIGUOUS`) rather than coercing them into arbitrary task execution.
- **Target Population**:
  - $N = 150$ newly curated natural language queries across 6 adversarial cohorts:
    1. *Adversarial Prompt Injections* ($N=25$): Instructions attempting to override system behavior (e.g., *"Ignore satellite sensors and output building boundaries"*).
    2. *Conflicting Multi-Task Requests* ($N=25$): Queries explicitly blending mutually incompatible tasks (e.g., *"Classify land cover and detect pixel changes between these two single images"*).
    3. *Polysemous Domain Jargon* ($N=25$): Vague industry terminology without task verbs (e.g., *"Provide geospatial situational awareness on this AOI"*).
    4. *Incomplete / Truncated Instructions* ($N=25$): Syntactically broken or single-word queries (e.g., *"Satellite Sentinel"*).
    5. *Negation Traps* ($N=25$): Requests specifying what *not* to do (e.g., *"Do not classify land cover, only observe"*).
    6. *Clear Valid In-Domain Controls* ($N=25$): Unambiguous baseline queries.
- **Evaluation Metrics**:
  - Interactive Clarification Recall and Precision.
  - False Dispatch Rate on Adversarial Prompts ($\text{FDR}_{\text{adv}}$).
  - Standardized Joint Selection Score ($s_{\text{joint}}$) distribution under adversarial input.
- **Pre-Registered Acceptance Criteria**:
  - Adversarial False Dispatch Rate $\text{FDR}_{\text{adv}} \le 10.0\%$ (at least $90\%$ of adversarial and conflicting queries intercepted via `CLARIFY_AMBIGUOUS` or rejected).

---

### Track 4.5: Adversarial Physical Metadata and Coordinate Perturbations

- **Hypothesis (H5)**: The closed-form geometric and radiometric checks in [`satquery/utils/geospatial.py`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/satquery/utils/geospatial.py) will reject $100\%$ of corrupted, un-coregistered, or non-finite payloads without throwing unhandled exceptions.
- **Target Population**:
  - $N = 120$ perturbed bitemporal and single-observation payloads across 8 machine-enforceable dimensions:
    - Coordinate Reference System (CRS) mismatch (EPSG:32634 vs EPSG:4326).
    - Sub-pixel grid translation exceeding registration tolerance ($\delta > 0.05\text{ px}$).
    - Non-orthogonal shear ($\kappa(\mathbf{M}) > 1.01$) and axis inversion ($\det(\mathbf{M}) < 0$).
    - Geometric rotation exceeding tolerance ($|\theta| > 0.28^\circ$).
    - Dimensional mismatch ($< 120 \times 120$ or mismatched bitemporal dimensions).
    - Temporal invalidity ($\Delta t < 5\text{ days}$ or $> 730\text{ days}$).
    - Non-finite scaling (NaN, Inf, raw DN $> 10,000$ unnormalized).
- **Evaluation Metrics**:
  - Violation Interception Recall ($100\%$ target).
  - Exact structured failure code matching (`ERR_CRS_MISMATCH`, `ERR_UNREGISTERED_GRID`, `ERR_CORRUPT_PAYLOAD`, etc.).
  - Zero unhandled crash exceptions (stability invariant).
- **Pre-Registered Acceptance Criteria**:
  - $\text{IDER}_{\text{obs}} = 0.00\%$ across all 120 perturbed payloads.
  - Zero unhandled exceptions ($100\%$ gracefully returned as structured `CompatibilityGateResult`).

---

## 4. Multiplicity Control & Statistical Analysis Plan

1. **Clustered Resampling**:
   - For all performance metrics on geographic transfer (Track 4.1), statistical uncertainty will be estimated using **Cluster Bootstrap** ($B=1,000$ replicates), resampling entire geographic tiles/cities as clusters to account for intra-scene spatial auto-correlation.
2. **Multiplicity Adjustment**:
   - Confirmatory hypothesis tests against baselines across multiple degradation regimes or query cohorts will apply the **Step-Down Holm procedure** at family-wise error rate $\alpha = 0.05$.
3. **Finite-Sample Safety Bounds**:
   - All zero-error rates on safety rejection tracks (4.2 and 4.5) will report both **Wilson score 95% confidence intervals** and **Clopper-Pearson exact binomial confidence intervals** to prevent unwarranted claims of universal safety.

---

## 5. Summary of Phase 4 Preregistration Status

| Protocol Item | Specification / Value | Status |
| :--- | :--- | :--- |
| **Model Weights** | Frozen from Phase 3 (ASL ResNet-50, FC-Siam-diff, TF-IDF + Logistic Regression) | **FROZEN** |
| **Routing Thresholds** | $\tau_{\text{intent}} = 0.65, \tau_{\text{margin}} = 0.15, \tau_{\text{sel}} = 1.0, T_{\text{cal}} = 0.0991$ | **FROZEN** |
| **Physical Gate Dimensions** | 8 dimensions in `satquery/utils/geospatial.py` | **FROZEN** |
| **Evaluation Mode** | Inference-only on local RTX 2050 (4 GB VRAM) | **COMMITTED** |
| **Confirmatory Experiments** | 5 Tracks (4.1 Geographic, 4.2 Cross-Sensor, 4.3 Atmospheric, 4.4 Linguistic, 4.5 Adversarial Metadata) | **PROPOSED** |
| **Protocol Status** | **DRAFT v1.0** — Ready for User Review and Formal Freeze | **PENDING USER FREEZE** |
