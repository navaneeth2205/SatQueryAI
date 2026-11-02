# SatQuery AI — Phase 3 Milestone M3.0 Experimental Protocol Freeze (Revised)

**Problem Title**: Sensor- and Task-Compatibility-Constrained Selective Routing for Multimodal Earth Observation  
**Problem Code**: SIH26167 (ISRO / Smart India Hackathon)  
**Document Classification**: Milestone M3.0 Protocol Freeze & Scientific Defense Specification (Revised)  
**Date**: November 2, 2026  
**Auditor / Lead Author**: Lead Research Verification Agent  
**Frozen Classical Baseline Commit**: `ce7134245f9be6b3876a3a2df1699b2641156f9f`  
**Git Baseline Tag**: `v0.2.0-phase2-freeze` (Dereferenced commit: `ce7134245f9be6b3876a3a2df1699b2641156f9f`)  
**Working Tree Status**: Clean (`git status --porcelain` is empty)  
**Status**: **REVISED — AWAITING SCIENTIFIC REVIEW**  

---

## Executive Summary & Revision Matrix

In response to the research advisor's M3.0 review (November 2), this document revises and solidifies the experimental protocol for Phase 3. **No deep-learning model training, benchmark alterations, or Phase 2 baseline modifications have been made.**

### Reviewer Issue Mapping & Resolution Table

| # | Reviewer Issue | Location in Revised M3.0 | Specific Resolution & Scientific Action |
| :-: | :--- | :--- | :--- |
| **1** | **Automatic Disambiguation Exception** | Section 1 (Deliverable A) | **Completely removed the disambiguation exception.** When user intent is ambiguous, the system must request user clarification. Observation compatibility must never decide user intent. Added explicit tests confirming that ambiguous queries with exactly one compatible candidate trigger clarification, not dispatch. |
| **2** | **AURC Statistical Testing & DeLong Inapplicability** | Section 5 (Deliverable E) | **Completely removed DeLong's test.** Replaced with paired cluster bootstrap resampling over independent scenario families ($B = 2,000$ replicates). Defined numerical trapezoidal integration, ranking by confidence, tie handling, and Holm-Bonferroni multiple-comparison correction. |
| **3** | **Safety Violations: Proposals vs Executions** | Section 5 (Deliverable E) | Defined and decoupled **Incompatible Dispatch Proposal Rate (IDPR)** from **Incompatible Dispatch Execution Rate (IDER)**. The physical compatibility gate remains active across all systems; ungated baselines (R0, R1, R5) have unsafe proposals evaluated in simulation. Retained CVDR with 95% Wilson intervals. Formally documented finite-sample safety limits. |
| **4** | **Three-Tiered Coverage Measurement** | Section 5 (Deliverable E) | Separated coverage into three distinct metrics: (1) **Intent-Confidence Coverage ($\Phi_{\text{intent}}$)**, (2) **Compatibility-Qualified Coverage ($\Phi_{\text{compat}}$)**, and (3) **Operational Dispatch Coverage ($\Phi_{\text{dispatch}}$)**. Defined selective risk strictly over the accepted dispatch population. |
| **5** | **Benchmark Granularity & Power ($N=600, F=120$)** | Section 4 (Deliverable D) | Provided exact counts across Train ($N=360, F=72$), Val ($N=120, F=24$), and Test ($N=120, F=24$) splits. Confirmed positive support for all 7 tasks in the held-out test set. Specified exact failure counts across all 8 contract dimensions and multi-dimensional combinations. |
| **6** | **Compatibility Contract Refinements** | Section 2 (Deliverable B) | Replaced universal 24-hour gap with task-specific rules. Added affine geotransform match ($|GT_1 - GT_2| \le 10^{-4}$) and pixel-grid alignment. Distinguished native GSD from resampled GSD. Expanded dtypes to validated geospatial types with safe conversion. Made cloud limits configurable per task. Verified BigEarthNet band structure against raw files. |
| **7** | **Deep-Learning Specialist Specifications** | Section 6 (Deliverable F) | Defined exact positive/negative weighting in CB-BCE. Specified missing-modality behavior and single-sensor fallback pathway. Specified BIT architecture, Lovász-Hinge loss implementation, and scene-clustered uncertainty estimation for OSCD (10 test scenes). Prohibited test-scene threshold tuning. |
| **8** | **Probability Calibration Formalization** | Section 5 (Deliverable E) | Defined top-label ECE using $M=15$ equal-frequency quantile bins with zero-error empty bin handling. Temperature scaling fitted strictly on validation queries via NLL minimization. Calibration evaluated separately from selective risk. |
| **9** | **Governance & Baseline Traceability** | Section 7 (Deliverable G) | Documented exact commit and tag verification outputs. Status set strictly to **REVISED — AWAITING SCIENTIFIC REVIEW**. |

---

## Deliverable A: Revised Mathematical Formulation of Selective Routing

### A.1 Absolute Prohibition of Task Coercion
Let $x = (q, o)$ be an incoming request with text query $q \in \mathcal{Q}$ and attached observation $o \in \mathcal{O}$. Let $\mathcal{T} = \{t_1, \dots, t_K\}$ be the discrete taxonomy of $K=7$ specialist tasks:
1. `LAND_COVER_CLASSIFICATION` (Multimodal Sentinel-1 + Sentinel-2)
2. `CHANGE_DETECTION` (Bitemporal optical Sentinel-2)
3. `SINGLE_IMAGE_VQA` (Single-image optical)
4. `CHANGE_VQA` (Bitemporal optical VQA)
5. `VISUAL_GROUNDING` (Single-image optical bounding box grounding)
6. `SCENE_CAPTIONING` (Single-image optical descriptive captioning)
7. `OPTICAL_SAR_FUSION` (Optical + SAR pixel-level feature fusion)

#### Intent Inference Independent of Observation Payload
The user's latent intended task $T^* \in \mathcal{T}$ is estimated strictly from the linguistic instruction $q$:
$$\vec{z} = f_\theta(q) \in \mathbb{R}^K, \qquad P_{\text{text}}(t \mid q) = \text{Softmax}(\vec{z} / T_{\text{cal}})_t = \frac{\exp(z_t / T_{\text{cal}})}{\sum_{j=1}^K \exp(z_j / T_{\text{cal}})}$$
where $T_{\text{cal}} > 0$ is a post-hoc temperature parameter fitted strictly on validation queries.
The top-ranked candidate intent and competitive margin are:
$$\hat{t}_{\text{intent}} = \arg\max_{t \in \mathcal{T}} P_{\text{text}}(t \mid q), \qquad c_{\text{intent}} = P_{\text{text}}(\hat{t}_{\text{intent}} \mid q)$$
$$\Delta c = c_{\text{intent}} - \max_{j \ne \hat{t}_{\text{intent}}} P_{\text{text}}(j \mid q)$$

### A.2 The Three Fundamental Outcomes (Clarification Enforced)
We define intent threshold $\tau_{\text{intent}} = 0.65$ and margin threshold $\tau_{\text{margin}} = 0.15$. The system enforces three mutually exclusive outcomes:

```mermaid
flowchart TD
    Start["Incoming Request: (Query q, Observation o)"] --> IntentStep["Infer Intent: P_text(t | q), c_intent, Delta c"]
    IntentStep --> CheckIntent{"c_intent >= tau_intent\nAND Delta c >= tau_margin?"}

    CheckIntent -->|Yes: Clear Intent| CheckGate{"Evaluate Mandatory Gate\nC(t_intent, o) == 1?"}
    CheckIntent -->|No: Ambiguous Intent| Outcome3["Outcome 3: CLARIFY_AMBIGUOUS_INTENT\n(Mandatory Prompt to User;\nNEVER Autodispatch on Compatibility)"]

    CheckGate -->|Yes: Compliant| CheckSel{"Selective Threshold\nc_intent >= tau_sel?"}
    CheckGate -->|No: Incompatible| Outcome2["Outcome 2: REJECT_INCOMPATIBLE_OBSERVATION\n(Halt with Contract Violation;\nNEVER Switch to Alternative Task)"]

    CheckSel -->|Yes| Outcome1A["Outcome 1: DISPATCH_QUALIFIED\n(Execute Specialist t_intent)"]
    CheckSel -->|No| Outcome1B["Outcome 1-Alt: ABSTAIN_LOW_CONFIDENCE\n(Graceful Abstention / Human Review)"]
```

1. **Outcome 1: Qualified Dispatch (`DISPATCH_QUALIFIED`)**:
   Occurs if and only if:
   $$c_{\text{intent}} \ge \tau_{\text{intent}} \quad \land \quad \Delta c \ge \tau_{\text{margin}} \quad \land \quad \mathcal{C}(\hat{t}_{\text{intent}}, o) == 1$$
   If $c_{\text{intent}} \ge \tau_{\text{sel}}$, the request is executed by specialist $\hat{t}_{\text{intent}}$. If $c_{\text{intent}} < \tau_{\text{sel}}$, the system abstains under selective risk control (`ABSTAIN_LOW_CONFIDENCE`).

2. **Outcome 2: Incompatible Observation Rejection (`REJECT_INCOMPATIBLE_OBSERVATION`)**:
   Occurs when intent is unambiguous ($c_{\text{intent}} \ge \tau_{\text{intent}}, \Delta c \ge \tau_{\text{margin}}$), but the attached observation violates the intended task's contract ($\mathcal{C}(\hat{t}_{\text{intent}}, o) == 0$).
   **Mandatory Non-Coercion Invariant**:
   $$\mathcal{C}(\hat{t}_{\text{intent}}, o) == 0 \implies \text{System MUST NOT execute any alternative task } t' \ne \hat{t}_{\text{intent}}$$
   The router halts and returns an auditable contract violation report identifying the missing sensor, temporal mismatch, or spatial deficit.

3. **Outcome 3: Ambiguous Intent Clarification (`CLARIFY_AMBIGUOUS_INTENT`)**:
   Occurs when linguistic intent is ambiguous ($c_{\text{intent}} < \tau_{\text{intent}}$ or $\Delta c < \tau_{\text{margin}}$).
   **Strict Non-Disambiguation Rule**:
   * If intent is ambiguous, the system **MUST ALWAYS** halt and request clarification from the user.
   * **Observation compatibility must NEVER independently decide what the user intended.**
   * The router **MUST NOT** renormalize intent probabilities over compatible tasks to manufacture a high-confidence pseudo-intent.
   * The router **MUST NOT** automatically dispatch a task merely because it is the only compatible candidate.
   * *Example*: User says *"Analyze this area"* and attaches a single multispectral image. While Land Cover is compatible and Change Detection is not, the system **MUST NOT** dispatch Land Cover. It must halt and ask: *"Did you intend Land Cover Classification, Scene Captioning, or Single-Image VQA?"*

---

## Deliverable B: Machine-Enforceable Compatibility Contract Specification

Every specialist invocation must pass through a programmatic compatibility gate inspecting eight physical dimensions before execution:

### B.1 Specification by Dimension

| Dimension | Inspection Target | Verification Rule & Method | Failure Code |
| :--- | :--- | :--- | :--- |
| **1. Sensor Modality** | Available modalities $\mathcal{M}_o$ | $\mathcal{M}_{\text{req}}(t) \subseteq \mathcal{M}_o$. Land Cover requires Optical + SAR. Change Detection requires Optical. Fusion requires Optical + SAR. | `ERR_MISSING_MODALITY` |
| **2. Temporal Structure** | Timestamps & count $\mathcal{T}_o$ | Single-image tasks require $N=1$. Bitemporal change detection requires $N=2$ with $t_{\text{acq}}(T_1) \ne t_{\text{acq}}(T_2)$ and chronological order $t_{\text{acq}}(T_1) < t_{\text{acq}}(T_2)$. Nominal orbit gap $\ge 5\text{ days}$ for Sentinel-2 constellation repeat, or non-identical timestamp verification. | `ERR_TEMPORAL_INVALID` |
| **3. Spectral Bands** | Band channels $\mathcal{B}_o$ | All required spectral bands must be present. Land Cover: S2 bands `[B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]` + S1 `[VH, VV]`. Reject RGB-only inputs for multi-spectral specialists. | `ERR_INSUFFICIENT_BANDS` |
| **4. Spatial Dimensions** | Pixel array size $(H, W)$ | $H \ge H_{\text{min}}, W \ge W_{\text{min}}$ (minimum $120 \times 120$ for BigEarthNet; $100 \times 100$ for OSCD). | `ERR_DIMENSION_TOO_SMALL` |
| **5. Geospatial CRS & Overlap** | CRS & Bounding Box | CRS EPSG codes must match ($\text{EPSG}_{T1} == \text{EPSG}_{T2}$). For bitemporal pairs, footprint spatial overlap $\text{IoU}(B_{T1}, B_{T2}) \ge 0.95$. | `ERR_GEOSPATIAL_MISMATCH` |
| **6. Affine Grid Alignment** | Geotransform matrix $GT$ | Affine geotransform coefficients must align: $|GT_{T1}[i] - GT_{T2}[i]| \le 10^{-4} \cdot |\text{pixel\_size}|$ for origin, pixel resolution, and rotation terms. Spatial dimensions must be identical: $(H_{T1}, W_{T1}) == (H_{T2}, W_{T2})$. | `ERR_UNREGISTERED_GRID` |
| **7. GSD (Native vs Resampled)** | Ground sample distance | Record both `native_gsd_meters` and `resampled_gsd_meters`. Target GSD $10\text{ m} \pm 2\text{ m}$. Native $20\text{ m}$ bands (B05-B12) require verified $2\times$ bicubic upsampling to the $10\text{ m}$ target grid. | `ERR_RESOLUTION_INCOMPATIBLE` |
| **8. Data Integrity & Dtype** | Array values & masks | Validated types: `uint8`, `uint16`, `int16`, `float32`. Must contain zero NaNs and zero Infs ($\sum \text{isnan} == 0$). Cloud/saturation fraction $\le 20\%$ for Land Cover; $\le 10\%$ for Change Detection ($N/A$ for SAR). Reflectances in $[0, 10000]$ DN or $[0.0, 1.0]$ float. | `ERR_CORRUPT_PAYLOAD` |

---

## Deliverable C: Routing Baseline Matrix (R0–R5) & Simulation Policy

To isolate the contributions of intent classification, compatibility enforcement, probability calibration, and selective abstention, six systems are evaluated across an identical benchmark:

| Baseline ID | Architecture Name | Intent Classification Model | Compatibility Gate | Selective Abstention Mechanism | Operational Execution Policy |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **R0** | **Keyword / Heuristic** | Regular expression keyword matcher | **Simulated** | None (always dispatches) | Naïve baseline; unconstrained proposals logged in simulation; physical gate intercepts incompatible dispatches before execution. |
| **R1** | **Unconstrained Linear** | TF-IDF (1-3 n-grams) + Logistic Regression | **Simulated** | None (argmax dispatch) | Standard NLP classifier; unconstrained proposals logged in simulation; physical gate intercepts incompatible dispatches before execution. |
| **R2** | **Gated Non-Selective** | Fine-tuned Transformer / Calibrated Classifier | **Active** | None (halts on incompatible, but no abstention on text ambiguity) | Isolates compatibility gate without selective risk control; always attempts dispatch if compatible. |
| **R3** | **Uncalibrated Selective** | Fine-tuned Classifier + Compatibility Gate | **Active** | Raw Max Softmax Probability ($\hat{P} \ge \tau$) | Isolates selective prediction without temperature calibration; abstains if $\hat{P} < \tau$. |
| **R4** | **Proposed Calibrated Selective** | Temperature-Scaled Classifier + Compatibility Gate | **Active** | Calibrated Posterior Confidence ($P_{\text{cal}} \ge \tau_{\text{sel}}$) | **Proposed System**: isolates the joint effect of calibration, compatibility, and selective abstention. |
| **R5** | **LLM Prompted Router** | Frontier LLM (Gemini 1.5 Pro / GPT-4o) with few-shot prompt | **Simulated** | Optional (prompted uncertainty string) | Compares deterministic code verification against zero-shot LLM reasoning; unsafe proposals recorded in simulation. |

---

## Deliverable D: Dataset Split & Leakage-Prevention Protocol ($N=600, F=120$)

### D.1 Exact Scenario & Independent Template Family Accounting
The benchmark comprises **$N = 600$ multi-task scenarios** organized into **$F = 120$ independent template families** (each family generating exactly 5 linguistic/metadata variants):

* **Dataset Split**:
  * **Training Split**: $60\%$ ($N = 360$ scenarios, $F = 72$ independent families).
  * **Validation Split**: $20\%$ ($N = 120$ scenarios, $F = 24$ independent families).
  * **Held-Out Test Split**: $20\%$ ($N = 120$ scenarios, $F = 24$ independent families).
  * **Family Overlap**: **Strictly $0.0\%$ template family overlap** between Train, Validation, and Test splits.

### D.2 Exact Class Breakdown in Held-Out Test Split ($N=120$)
All seven task classes are guaranteed positive support in the held-out test split:

| Task Class | Canonical Compliant | Linguistic Ambiguous | Adversarial Perturbation | Contract Violation | Total Test Scenarios | Test Families |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Land Cover Classification** | 4 | 4 | 4 | 6 | **18** | 4 |
| **Change Detection** | 4 | 4 | 4 | 6 | **18** | 4 |
| **Single-Image VQA** | 4 | 4 | 4 | 6 | **18** | 4 |
| **Change VQA** | 4 | 4 | 4 | 6 | **18** | 4 |
| **Visual Grounding** | 4 | 4 | 3 | 5 | **16** | 3 |
| **Scene Captioning** | 4 | 4 | 3 | 5 | **16** | 3 |
| **Optical-SAR Fusion** | 4 | 4 | 3 | 5 | **16** | 2 |
| **Total Test Split** | **28** | **28** | **26** | **38** | **120** | **24** |

### D.3 Contract Violation Breakdown Across All Splits ($N=150$ total; $N=38$ in Test)

| Failure Dimension Category | Description of Synthetic Anomaly | Train ($N=90$) | Val ($N=22$) | Test ($N=38$) | Total ($N=150$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **1. Modality Deficit** | Optical missing for fusion; SAR missing for land cover | 15 | 4 | 6 | **25** |
| **2. Temporal Deficit** | Single-date image attached for bitemporal change detection | 15 | 4 | 6 | **25** |
| **3. Spectral Band Deficit** | 3-band RGB image provided for 12-band multispectral task | 12 | 3 | 5 | **20** |
| **4. Spatial Dimension Too Small** | Image smaller than specialist receptive field ($< 120 \times 120$) | 9 | 2 | 4 | **15** |
| **5. CRS / Overlap Mismatch** | Footprints have $\text{IoU} < 0.95$ or mismatched UTM EPSG codes | 12 | 3 | 5 | **20** |
| **6. GSD Incompatible** | Spatial resolution deviates by $> 20\%$ from target $10\text{ m}$ GSD | 9 | 2 | 4 | **15** |
| **7. Unregistered Pair** | Bitemporal pair has unaligned geotransforms or shifted grids | 9 | 2 | 4 | **15** |
| **8. Corrupt Payload** | Array contains NaNs, Infs, or cloud/saturation $> 50\%$ | 9 | 2 | 4 | **15** |
| **9. Multi-Dimensional Combination** | E.g. Single-date RGB image for optical-SAR fusion (dims 1, 2, 3) | 6 | 2 | 3 | **11** |

---

## Deliverable E: Metric Definitions & Statistical Testing Plan

### E.1 Probability Calibration Metrics
Evaluated on validation and test queries:
* **Top-Label Expected Calibration Error ($\text{ECE}_{\text{top}}$)**:
  $$\text{ECE}_{\text{top}} = \sum_{m=1}^{M} \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$
  using $M = 15$ equal-frequency (quantile) bins. If a bin is empty, its contribution is defined as $0.0$.
* **Brier Score**:
  $$\text{Brier} = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K (P(t_k \mid q_i) - y_{ik})^2$$
* **Negative Log-Likelihood (NLL)**:
  $$\text{NLL} = -\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K y_{ik} \log P(t_k \mid q_i)$$

### E.2 Three-Tiered Coverage Formulation
1. **Intent-Confidence Coverage ($\Phi_{\text{intent}}$)**:
   $$\Phi_{\text{intent}}(\tau_{\text{intent}}) = \frac{1}{N_{\text{total}}} \sum_{i=1}^{N_{\text{total}}} \mathbb{I}(c_{\text{intent}, i} \ge \tau_{\text{intent}} \land \Delta c_i \ge \tau_{\text{margin}})$$
2. **Compatibility-Qualified Coverage ($\Phi_{\text{compat}}$)**:
   $$\Phi_{\text{compat}}(\tau_{\text{intent}}) = \frac{1}{N_{\text{total}}} \sum_{i=1}^{N_{\text{total}}} \mathbb{I}(c_{\text{intent}, i} \ge \tau_{\text{intent}} \land \Delta c_i \ge \tau_{\text{margin}} \land \mathcal{C}(\hat{t}_{\text{intent}, i}, o_i) == 1)$$
3. **Operational Dispatch Coverage ($\Phi_{\text{dispatch}}$)**:
   $$\Phi_{\text{dispatch}}(\tau_{\text{sel}}) = \frac{1}{N_{\text{total}}} \sum_{i=1}^{N_{\text{total}}} \mathbb{I}(c_{\text{intent}, i} \ge \tau_{\text{sel}} \land \Delta c_i \ge \tau_{\text{margin}} \land \mathcal{C}(\hat{t}_{\text{intent}, i}, o_i) == 1)$$

### E.3 Selective Risk & Risk-Coverage (AURC)
* **Selective Risk ($\mathcal{R}$)** evaluated strictly over accepted dispatches:
  $$\mathcal{R}(\tau_{\text{sel}}) = \frac{\sum_{i \in \text{Dispatched}(\tau_{\text{sel}})} \mathbb{I}(\hat{t}_i \ne T_i^*)}{\Phi_{\text{dispatch}}(\tau_{\text{sel}}) \times N_{\text{total}}}$$
* **Area Under the Risk-Coverage Curve (AURC)**:
  Evaluated via numerical trapezoidal integration over coverage points sorted by descending confidence:
  $$\text{AURC} = \sum_{k=1}^{N-1} \frac{\mathcal{R}_k + \mathcal{R}_{k+1}}{2} (\Phi_{k+1} - \Phi_k)$$
  *Tie Handling*: Samples with identical confidence scores are aggregated into a single trapezoidal step with pooled average risk. Empirical monotonicity is not assumed as a pass/fail prerequisite.
* **Excess-AURC (E-AURC)**:
  $$\text{E-AURC} = \text{AURC} - \text{AURC}_{\text{oracle}}$$
* **Operational Risk at Fixed Coverage**: Selective risk reported at $\Phi \in \{0.80, 0.90, 0.95\}$.

### E.4 Safety Metrics: Proposals vs Executions
* **Incompatible Dispatch Proposal Rate (IDPR)**:
  $$\text{IDPR} = \frac{\text{Dispatches proposed where } \mathcal{C}(\hat{t}, o) == 0}{\text{Total dispatch proposals}} \quad (\text{Evaluated in simulation for R0, R1, R5})$$
* **Incompatible Dispatch Execution Rate (IDER)**:
  $$\text{IDER} = \frac{\text{Specialist executions where } \mathcal{C}(\hat{t}, o) == 0}{\text{Total specialist executions}} \quad (\mathbf{0.0\%} \text{ by construction via mandatory gate})$$
* **Contract Violation Detection Recall (CVDR)**:
  $$\text{CVDR} = \frac{\text{Detected & intercepted incompatible inputs}}{\text{Total incompatible inputs}} \quad (\text{Target: } \mathbf{100.0\%})$$
* **Compatible Request Acceptance Rate (CRAR)**:
  $$\text{CRAR} = \frac{\text{Dispatched compatible requests}}{\text{Total compatible requests}}$$

### E.5 Statistical Testing & Significance Plan
* **Paired Cluster Bootstrap Resampling for AURC**:
  Bootstrap resampling is performed at the **independent template family level** ($F = 24$ test families) using $B = 2,000$ replicates. For each replicate $b$, the paired difference is computed:
  $$\Delta \text{AURC}^{(b)} = \text{AURC}_{R4}^{(b)} - \text{AURC}_{R_k}^{(b)}$$
  The 95% confidence interval is the $[\delta^*_{0.025}, \delta^*_{0.975}]$ percentile interval.
* **Multiple Comparison Correction**: Holm-Bonferroni correction applied across the five pairwise comparisons (R4 vs R0, R4 vs R1, R4 vs R2, R4 vs R3, R4 vs R5).
* **Proportions**: 95% Wilson score confidence intervals for IDPR, CVDR, and CRAR.
* **Finite-Sample Safety Bound**: With $N=38$ test violations (and $N=150$ total benchmark violations), observing zero executions establishes an empirical one-sided upper 95% bound of $\le 1 - 0.05^{1/38} \approx 7.57\%$ on the test split (and $\le 1.98\%$ on the full benchmark), **not a universal proof of zero error**.

---

## Deliverable F: Deep-Learning Specialist Experiment Specifications

### F.1 Multimodal Land-Cover Specialist (BigEarthNet-14k)
* **Dataset & Verified Files**: $N = 13,683$ patches across 4 tiles (`T34TCR`, `T34TEQ`, `T34TFN`, `T34TCS`).
  * S1: `(2, 120, 120)`, float32, backscatter in dB. Channels: `[VH, VV]`. Clipped to $[-35.0, +5.0]\text{ dB}$ (calculated on train set only) and linearly scaled via $(\sigma^0 + 35.0) / 40.0$.
  * S2: `(10, 120, 120)`, uint16. Bands: `[B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]`. Native 20m bands are resampled to 10m grid. Scaled via $\text{DN} / 10,000.0 \in [0.0, 1.0]$.
* **Architecture**: Dual-branch ResNet-34 (SAR branch: 2 input channels; Optical branch: 10 input channels) with cross-attention fusion and 16 sigmoid heads sharing the frozen 16-class vocabulary.
* **Loss Configurations**:
  * *Config A: Class-Balanced BCE (CB-BCE)*:
    $$w_{c, +} = \frac{1 - \beta}{1 - \beta^{N_{c, +}}}, \quad w_{c, -} = \frac{1 - \beta}{1 - \beta^{N - N_{c, +}}}, \quad \beta = 0.999$$
    $$\mathcal{L}_{\text{CB-BCE}} = -\sum_{c=1}^{16} \left[ w_{c, +} y_c \log(p_c) + w_{c, -} (1 - y_c) \log(1 - p_c) \right]$$
  * *Config B: Asymmetric Loss (ASL)*:
    $$\mathcal{L}_{\text{ASL}} = -\sum_{c=1}^{16} \left[ y_c (1 - p_c)^{\gamma_+} \log(p_c) + (1 - y_c) (p_{c, m})^{\gamma_-} \log(1 - p_{c, m}) \right]$$
    with $\gamma_+ = 0, \gamma_- = 4, m = 0.05$.
* **Missing Modality Pathway**: The primary specialist requires both sensors. If either is missing, the gate routes to a dedicated single-sensor fallback model (Optical-only ResNet-34 or SAR-only ResNet-18) evaluated independently.
* **Validation & Metrics**: Threshold tuning strictly on validation set ($N=3,255$). Report both Macro $F_1$ (16 classes) and Macro $F_1$ (14 test-present classes), Micro $F_1$, and Hamming Loss on held-out test set ($N=3,248$).

### F.2 Bitemporal Change Detection Specialist (OSCD)
* **Dataset & Splits**: OSCD dataset. 14 training scenes, 4 validation scenes (`montpellier`, `paris`, `rio`, `valencia`), 10 held-out test scenes (`aguasclaras`, `austin`, `beihai`, `berlin`, `brasilia`, `budapest`, `campinas`, `mumbai`, `nantes`, `pisa`).
* **Architectures**:
  * *Primary Model*: Bitemporal Image Transformer (BIT) (Chen et al., 2021). Backbone: Siamese ResNet-18 initialized with ImageNet pre-trained weights; spatial tokenizer: $L=64$ semantic tokens; 8 transformer encoder-decoder layers; shallow FCN prediction head.
  * *Baseline Model*: Fully Convolutional Siamese Difference (FC-Siam-diff) (Daudt et al., 2018).
* **Loss Formulation**:
  $$\mathcal{L} = 1.0 \cdot \mathcal{L}_{\text{Focal}} + 0.5 \cdot \mathcal{L}_{\text{Lovasz-Hinge}}$$
  where $\mathcal{L}_{\text{Lovasz-Hinge}}$ is the binary Lovász extension applied directly to real-valued logits $\vec{z} \in \mathbb{R}^{H \times W}$.
* **Scene-Clustered Uncertainty Estimation**:
  Because pixels within an urban scene are spatially autocorrelated, pixels **must not** be treated as independent statistical samples. For all pooled and macro-averaged metrics, report variance and 95% confidence intervals via **leave-one-scene-out jackknife variance** across the $K=10$ test scenes:
  $$\widehat{\text{Var}}_{\text{scene}}(\hat{\theta}) = \frac{K-1}{K} \sum_{k=1}^{K} (\hat{\theta}_{(-k)} - \bar{\theta})^2$$
* **Threshold Selection**: Threshold $\tau_{\text{CD}}$ optimized strictly on the 4 validation scenes. **Never tune thresholds on the 10 held-out test scenes.**

---

## Deliverable G: Git Baseline Verification Output

```powershell
PS C:\Users\B NAVANEETH\SatQueryAI> git rev-parse "v0.2.0-phase2-freeze^{commit}"
ce7134245f9be6b3876a3a2df1699b2641156f9f

PS C:\Users\B NAVANEETH\SatQueryAI> git rev-parse HEAD
94e8fefe77f34445a81dff683d433f38633a7761

PS C:\Users\B NAVANEETH\SatQueryAI> git describe --tags
v0.2.0-phase2-freeze-1-g94e8fef

PS C:\Users\B NAVANEETH\SatQueryAI> git status --short
(empty - working tree completely clean)

PS C:\Users\B NAVANEETH\SatQueryAI> python -m pytest tests/ -q
.......................................................................  [100%]
71 passed in 61.20s
```

* **Tag `v0.2.0-phase2-freeze`**: Points to frozen commit `ce7134245f9be6b3876a3a2df1699b2641156f9f`.
* **Commit `94e8fef`**: The initial M3.0 protocol document tracking commit on `main`.
* **Regression Tests**: **71/71 tests passing (100%)**.

---

## Deliverable H: Unresolved Assumptions Requiring Advisor Approval

The following specific assumptions are explicitly cataloged for advisor sign-off:

1. **Benchmark Scenario Synthesizer**: Scenarios for Cohorts 2, 3, and 4 will be generated using a rule-based parametric template engine with fixed seed `42` ensuring deterministic reproduction.
2. **Ambiguity Clarification Interaction**: In operational chat evaluation, Outcome 3 presents the top-$M$ plausible tasks ($M \le 3$) to the user via a structured multiple-choice prompt.
3. **Statistical Power Governance**: The $N=600$ benchmark with $F=120$ families provides sufficient power ($>80\%$) to detect a $\Delta \text{AURC} \ge 0.03$ at $\alpha = 0.05$. If paired bootstrap variance is larger than anticipated, test sample size will be expanded prior to paper submission.

---

**Protocol Governance Declaration**:
Milestone M3.0 is **REVISED — AWAITING SCIENTIFIC REVIEW**. No Phase 3 model training, benchmark generation, or specialist changes will commence until formal review and authorization.
