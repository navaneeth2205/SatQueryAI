# SatQuery AI — Phase 3 Milestone M3.0 Experimental Protocol Freeze (Final Research Specification)

**Problem Title**: Sensor- and Task-Compatibility-Constrained Selective Routing for Multimodal Earth Observation  
**Problem Code**: SIH26167 (ISRO / Smart India Hackathon)  
**Document Classification**: Milestone M3.0 Protocol Freeze & Scientific Defense Specification (Final Research Specification)  
**Date**: November 2, 2026  
**Auditor / Lead Author**: Lead Research Verification Agent  
**Frozen Classical Baseline Commit**: `ce7134245f9be6b3876a3a2df1699b2641156f9f`  
**Git Baseline Tag**: `v0.2.0-phase2-freeze` (Dereferenced commit: `ce7134245f9be6b3876a3a2df1699b2641156f9f`)  
**Working Tree Status**: Clean (`git status --porcelain` is empty)  
**Status**: **REVISED — AWAITING SCIENTIFIC REVIEW**  

---

## Executive Summary & Revision Matrix

This document provides the final, reconciled scientific protocol for Milestone M3.0 of SatQuery AI. It resolves all methodological, mathematical, and publication-level issues raised in the research review.

**Research Governance Invariant**: **No deep-learning models have been trained, no benchmark scenario files generated, and the frozen Phase 2 baseline tag remains strictly unchanged.** Authorization for Milestone M3.1 remains blocked pending scientific review and sign-off on this protocol.

### Comprehensive Reviewer Issue Resolution Table

| # | Priority Area | Specific Reviewer Finding | Protocol Resolution & Scientific Correction | Section |
| :-: | :--- | :--- | :--- | :--- |
| **1** | **Statistical Power Validity** | Unreconciled paired variance ($\sigma_\Delta \approx 0.06$ vs 0.04/0.08); Holm step-down confounded with first-step $\alpha=0.01$. | Provided complete simulation script (`tests/power_analysis_reproducible.py`, seed `42`). Formulated bivariate model with individual $\sigma = 0.08$ and cross-system correlation $\rho \in \{0.875, 0.719, 0.500\}$ mapping directly to $\sigma_\Delta \in \{0.04, 0.06, 0.08\}$. Separately reported individual two-sided power ($\alpha=0.05$), conservative Bonferroni tier-1 ($\alpha=0.01$), and complete 5-comparison step-down Holm procedure. | Deliverable E.5 |
| **2** | **Baseline Ablation Isolation** | R2 vs R3 changed both calibration and selective abstention simultaneously; R4 vs R5 confounded deterministic routing with LLM architecture. | Redesigned into a clean 2x2 factorial ablation grid: `R1` vs `R1-Gated` (isolates compatibility gate); `R2` vs `R2-Cal` (isolates calibration alone); `R2` vs `R3` (isolates selective abstention alone); `R3` vs `R4` (isolates calibration under abstention); `R2-Cal` vs `R4` (isolates abstention under calibrated scores). Evaluated R5 separately as an external zero-shot LLM baseline without claiming single-factor ablation. | Deliverable C.1–C.2 |
| **3** | **Affine Geospatial Mathematics** | Grid-phase assumed north-up unrotated rasters; GSD check ignored rotation terms; conflicting resolution tolerances ($10\pm2\text{ m}$ vs 1%). | Implemented full 6-parameter affine coordinate transformations $\mathbf{A} \vec{x} + \vec{t}$. Reconciled tolerances: $10\pm2\text{ m}$ is the single-sensor operational acceptance window; $\le 1\%$ is the inter-raster relative scale co-registration tolerance. Defined shortest distance to nearest integer pixel phase. Explicitly labeled $0.05\text{ px}$ threshold as an implementation-selected engineering constraint with literature justification. | Deliverable B.2 |
| **4** | **Independent Benchmark Ground Truth** | $V_3$ ambiguity was defined circularly via router confidence scores ($c < 0.65$ or $\Delta c < 0.15$). | Eliminated model-dependent ambiguity. Formulated an independent dual-annotator human linguistic protocol for ambiguity ($A^* \in \{0, 1\}$) with Cohen's $\kappa \ge 0.85$. Target action for ambiguous queries defined as $T^* = \text{CLARIFY}$. Router confidence is evaluated *against* human labels. Audited $V_4$ adversarial perturbations via human review. | Deliverable D.1–D.2 |
| **5** | **Per-Dimension Safety Evaluation** | 28 held-out test violations were not allocated across the 8 physical dimensions. | Added explicit fixed allocation table distributing the $N=28$ test violations across all 8 dimensions (3–4 violations per dimension). Documented finite-sample safety bound: zero failures in 28 tests yields a one-sided upper 95% Wilson bound of $9.68\%$, not a proof of zero population risk. | Deliverable D.4, E.4 |
| **6** | **Protocol Consistency** | Clarified threshold status; detailed AURC code implementation; justified temporal/cloud limits. | Designated $\tau_{\text{intent}}=0.65, \tau_{\text{margin}}=0.15$ as initial nominal baselines formally tuned via 2D validation grid search. Documented exact Python logic for AURC tie handling, empty populations, and horizontal left-continuation $\mathcal{R}(0) := \mathcal{R}_1$. Explicitly identified temporal/cloud limits as benchmark-specific operational contracts. | Deliverables B.3, E.1–E.3 |
| **7** | **Novelty & Prior Art Matrix** | Novelty claims ("first", "never integrated") were unsubstantiated against existing EO assistants. | Built a comprehensive related-work capability matrix evaluating Earth-GPT, GeoChat, SkyEye, RS-LLaVA, Learning to Route, SelectiveNet, and STAC validators across 6 specific capabilities. Framed SatQuery AI's contribution strictly as a **scientific research hypothesis to be tested**. | Deliverable H |
| **8** | **Model Implementation Reproducibility** | Missing optimizer, scheduler, seeds, and training budgets for deep-learning specialists. | Provided complete hyperparameter and training recipes for BigEarthNet (AdamW, cosine annealing, batch 64, 40 epochs, patience 7, 3 seeds) and OSCD BIT (Chen et al. 2021, Siamese ResNet-18, 4 enc + 4 dec transformer layers, shallow FCN, Focal + Lovász-Hinge loss). Explicitly noted scope: 2 implemented specialists, 5 routing/safety contract targets. | Deliverable F.1–F.2 |

---

## Deliverable A: Revised Mathematical Formulation of Selective Routing

### A.1 Absolute Prohibition of Task Coercion
Let $x = (q, o)$ be an incoming query-observation tuple, where $q \in \mathcal{Q}$ represents the user's natural language instruction and $o \in \mathcal{O}$ represents the attached Earth Observation payload metadata. Let $\mathcal{T} = \{t_1, \dots, t_K\}$ be the discrete taxonomy of $K=7$ specialist tasks:
1. `LAND_COVER_CLASSIFICATION` (Multimodal Sentinel-1 SAR + Sentinel-2 MSI)
2. `CHANGE_DETECTION` (Bitemporal optical Sentinel-2)
3. `SINGLE_IMAGE_VQA` (Single-image optical)
4. `CHANGE_VQA` (Bitemporal optical VQA)
5. `VISUAL_GROUNDING` (Single-image optical bounding box grounding)
6. `SCENE_CAPTIONING` (Single-image optical descriptive captioning)
7. `OPTICAL_SAR_FUSION` (Optical + SAR pixel-level feature fusion)

#### Intent Inference Independent of Observation Payload
The user's latent intended task $T^* \in \mathcal{T}$ is inferred strictly from the linguistic instruction $q$:
$$\vec{z} = f_\theta(q) \in \mathbb{R}^K, \qquad P_{\text{text}}(t \mid q) = \text{Softmax}(\vec{z} / T_{\text{cal}})_t = \frac{\exp(z_t / T_{\text{cal}})}{\sum_{j=1}^K \exp(z_j / T_{\text{cal}})}$$
where $T_{\text{cal}} > 0$ is a post-hoc calibration temperature fitted strictly on validation queries via negative log-likelihood minimization.

The top-ranked candidate intent and competitive confidence margin are:
$$\hat{t}_{\text{intent}} = \arg\max_{t \in \mathcal{T}} P_{\text{text}}(t \mid q), \qquad c_{\text{intent}} = P_{\text{text}}(\hat{t}_{\text{intent}} \mid q)$$
$$\Delta c = c_{\text{intent}} - \max_{j \ne \hat{t}_{\text{intent}}} P_{\text{text}}(j \mid q)$$

### A.2 The Three Fundamental Execution Pathways
With nominal intent threshold $\tau_{\text{intent}} = 0.65$ and margin threshold $\tau_{\text{margin}} = 0.15$ (tuned on validation split, Section E.1), the system enforces three mutually exclusive execution pathways:

```mermaid
flowchart TD
    Start["Incoming Request: (Query q, Observation o)"] --> IntentStep["Infer Intent: P_text(t | q), c_intent, Delta c"]
    IntentStep --> CheckIntent{"c_intent >= tau_intent\nAND Delta c >= tau_margin?"}

    CheckIntent -->|Yes: Unambiguous Intent| CheckGate{"Evaluate Mandatory Gate\nC(t_intent, o) == 1?"}
    CheckIntent -->|No: Ambiguous Intent| Outcome3["Outcome 3: CLARIFY_AMBIGUOUS_INTENT\n(Mandatory Prompt to User;\nNEVER Autodispatch on Compatibility)"]

    CheckGate -->|Yes: Compliant Payload| CheckSel{"Selective Threshold\nc_intent >= tau_eff?"}
    CheckGate -->|No: Incompatible Payload| Outcome2["Outcome 2: REJECT_INCOMPATIBLE_OBSERVATION\n(Halt with Contract Violation;\nNEVER Coerce to Alternative Task)"]

    CheckSel -->|Yes| Outcome1A["Outcome 1: DISPATCH_QUALIFIED\n(Execute Specialist t_intent)"]
    CheckSel -->|No| Outcome1B["Outcome 1-Alt: ABSTAIN_LOW_CONFIDENCE\n(Graceful Abstention / Human Review)"]
```

1. **Outcome 1: Qualified Dispatch (`DISPATCH_QUALIFIED`)**:
   Occurs if and only if:
   $$c_{\text{intent}} \ge \tau_{\text{intent}} \quad \land \quad \Delta c \ge \tau_{\text{margin}} \quad \land \quad \mathcal{C}(\hat{t}_{\text{intent}}, o) == 1$$
   If $c_{\text{intent}} \ge \tau_{\text{sel}}$, the request is executed by specialist $\hat{t}_{\text{intent}}$. If $c_{\text{intent}} < \tau_{\text{sel}}$, the system abstains under selective risk control (`ABSTAIN_LOW_CONFIDENCE`).

2. **Outcome 2: Incompatible Observation Rejection (`REJECT_INCOMPATIBLE_OBSERVATION`)**:
   Occurs when linguistic intent is unambiguous ($c_{\text{intent}} \ge \tau_{\text{intent}}, \Delta c \ge \tau_{\text{margin}}$), but the attached observation violates the intended task's contract ($\mathcal{C}(\hat{t}_{\text{intent}}, o) == 0$).
   $$\mathcal{C}(\hat{t}_{\text{intent}}, o) == 0 \implies \text{System MUST NOT execute any alternative task } t' \ne \hat{t}_{\text{intent}}$$
   The router halts execution and returns a structured diagnostic report detailing the contract violation code.

3. **Outcome 3: Ambiguous Intent Clarification (`CLARIFY_AMBIGUOUS_INTENT`)**:
   Occurs when linguistic intent is ambiguous ($c_{\text{intent}} < \tau_{\text{intent}}$ or $\Delta c < \tau_{\text{margin}}$).
   * The system **MUST ALWAYS** halt and request clarification from the user.
   * **Observation compatibility must NEVER independently decide or coerce user intent.**
   * The router **MUST NOT** renormalize intent probabilities over compatible candidates to manufacture an artificial dispatch.
   * The router **MUST NOT** automatically dispatch a task merely because it is the sole compatible candidate.

---

## Deliverable B: Machine-Enforceable Compatibility Contract Specification

Every specialist execution request must satisfy the programmatic compatibility gate $\mathcal{C}(t, o) \in \{0, 1\}$ inspecting eight physical dimensions:

### B.1 Specification by Physical Dimension

| Dimension | Inspection Target | Verification Rule & Method | Failure Code |
| :--- | :--- | :--- | :--- |
| **1. Sensor Modality** | Available modalities $\mathcal{M}_o$ | $\mathcal{M}_{\text{req}}(t) \subseteq \mathcal{M}_o$. Land Cover requires Optical + SAR. Change Detection requires Optical. Fusion requires Optical + SAR. | `ERR_MISSING_MODALITY` |
| **2. Temporal Structure** | Timestamps & count $\mathcal{T}_o$ | Single-image tasks require $N=1$. Change Detection requires $N=2$ with $t_{\text{acq}}(T_1) < t_{\text{acq}}(T_2)$ and $\Delta t \ge 5\text{ days}$. Fusion requires $\Delta t \le 72\text{ hours}$ (dynamic) or $\le 10\text{ days}$ (static). | `ERR_TEMPORAL_INVALID` |
| **3. Spectral Bands** | Band channels $\mathcal{B}_o$ | Required bands present. Land Cover: S2 `[B02-B08, B8A, B11, B12]` + S1 `[VH, VV]`. Reject RGB-only for multispectral specialists. | `ERR_INSUFFICIENT_BANDS` |
| **4. Spatial Dimensions** | Pixel array size $(H, W)$ | $H \ge H_{\text{min}}, W \ge W_{\text{min}}$ (min $120 \times 120$ for BigEarthNet; min $100 \times 100$ for OSCD). | `ERR_DIMENSION_TOO_SMALL` |
| **5. Geospatial CRS & Reprojection** | CRS Authority | CRS inspection. If $\text{EPSG}_{1} \ne \text{EPSG}_{2}$, automated reprojection into UTM target CRS is evaluated. Rejection occurs only if reprojection fails or distortion exceeds limits. | `ERR_GEOSPATIAL_MISMATCH` |
| **6. Affine Grid Alignment** | Geotransform matrix $GT$ | Executable 6-step alignment verification (Section B.2). Sub-pixel origin phase offset $\le 0.05\text{ px}$. | `ERR_UNREGISTERED_GRID` |
| **7. GSD (Native vs Resampled)** | Ground sample distance | Operational range $10\pm2\text{ m}$. Multi-image scale consistency $\le 1\%$. Upsampling provenance logged. | `ERR_RESOLUTION_INCOMPATIBLE` |
| **8. Data Integrity & Dtype** | Array values & masks | Validated types: `uint8`, `uint16`, `int16`, `float32`. Zero NaNs and Infs ($\sum \text{isnan} == 0$). Cloud fraction limits enforced (Section B.3). | `ERR_CORRUPT_PAYLOAD` |

### B.2 Full Affine Geospatial Alignment Verification Algorithm
Let $I_1$ and $I_2$ be two geospatial raster payloads with 6-parameter affine geotransforms $GT_1, GT_2$:
$$\begin{pmatrix} X_{\text{geo}} \\ Y_{\text{geo}} \end{pmatrix} = \mathbf{t}_k + \mathbf{A}_k \begin{pmatrix} x_k \\ y_k \end{pmatrix} = \begin{pmatrix} GT_k[0] \\ GT_k[3] \end{pmatrix} + \begin{pmatrix} GT_k[1] & GT_k[2] \\ GT_k[4] & GT_k[5] \end{pmatrix} \begin{pmatrix} x_k \\ y_k \end{pmatrix}$$

The physical compatibility gate executes the following sequence:

1. **Step 1: Common Projected CRS Resolution**:
   Inspect $\text{CRS}_1$ and $\text{CRS}_2$. If $\text{EPSG}_1 \ne \text{EPSG}_2$, the gate invokes GDAL/rasterio warp to reproject $I_2$ into the primary raster's UTM coordinate system. Mismatched EPSG codes do not automatically trigger failure; rejection occurs only if reprojection fails or changes ground pixel area by $> 5\%$.
2. **Step 2: Reconciling the Two Resolution Tolerances**:
   * **Rule A: Single-Raster Operational Specialist Receptive Window ($10 \pm 2\text{ m}$)**:
     The ground sampling distance of each raster along rows and columns, accounting for rotation, is:
     $$s_{x, k} = \sqrt{GT_k[1]^2 + GT_k[4]^2}, \qquad s_{y, k} = \sqrt{GT_k[2]^2 + GT_k[5]^2}$$
     Both $s_{x, k}$ and $s_{y, k}$ must fall within $[8.0\text{ m}, 12.0\text{ m}]$ for standard specialist receptive compatibility.
   * **Rule B: Inter-Raster Relative Scale Co-Registration Tolerance ($\le 1\%$)**:
     In multi-image tasks (change detection, multimodal optical-SAR fusion), the two rasters must share a matching pixel scale:
     $$\frac{|s_{x, 1} - s_{x, 2}|}{s_{x, 1}} \le 0.01 \quad \text{and} \quad \frac{|s_{y, 1} - s_{y, 2}|}{s_{y, 1}} \le 0.01$$
     Native 20m Sentinel-2 bands require verified $2\times$ bicubic upsampling to the 10m grid prior to input.
3. **Step 3: Relative Grid Displacement in Continuous Pixel Coordinates**:
   To test whether raster 2 aligns with raster 1's discrete pixel grid, project raster 2's origin $\mathbf{t}_2$ into raster 1's continuous pixel coordinate space:
   $$\begin{pmatrix} x_{1}^{\text{proj}} \\ y_{1}^{\text{proj}} \end{pmatrix} = \mathbf{A}_1^{-1} \left( \mathbf{t}_2 - \mathbf{t}_1 \right)$$
   Compute the shortest signed Euclidean distance to the nearest integer pixel center:
   $$\delta x = x_{1}^{\text{proj}} - \text{round}(x_{1}^{\text{proj}}), \qquad \delta y = y_{1}^{\text{proj}} - \text{round}(y_{1}^{\text{proj}})$$
   $$\delta_{\text{phase}} = \sqrt{\delta x^2 + \delta y^2} \quad \text{pixels}$$
   Also verify grid orientation match: $|\theta_{x, 1} - \theta_{x, 2}| \le 0.1^\circ$ where $\theta_x = \arctan2(GT[4], GT[1])$.
4. **Step 4: Implementation Engineering Constraint Justification ($\delta_{\text{phase}} \le 0.05\text{ px}$)**:
   The threshold $\epsilon_{\text{phase}} \le 0.05\text{ px}$ is an **implementation-selected engineering constraint**. In bi-temporal change detection, sub-pixel misregistration produces spurious differential edges along boundaries. As established in remote-sensing coregistration sensitivity literature (Dai & Khorram, 1998; Townshend et al., 1992), registration errors exceeding $0.1$ to $0.2$ pixels can generate up to a $10\%-15\%$ degradation in change detection accuracy in structured urban regions. SatQuery AI adopts $\epsilon_{\text{phase}} \le 0.05\text{ px}$ ($0.5\text{ m}$ on a 10m grid) as an engineering safety margin to ensure that displacement remains within the sensor point spread function. Displacements exceeding $0.05\text{ px}$ trigger sub-pixel coregistration prior to specialist execution.
5. **Step 5: Valid-Data Pixel Intersection Check**:
   Replaces bounding-box IoU. Let $M_1(x, y), M_2(x, y) \in \{0, 1\}$ be valid-data masks (excluding NoData, swath edges, and saturated pixels). The valid intersection ratio is:
   $$\Omega_{\text{valid}} = \frac{\sum_{x, y} M_1(x, y) \cdot M_2(x, y)}{\min\left(\sum M_1, \sum M_2\right)} \ge 0.95$$
   This ensures that $\ge 95\%$ of the evaluated spatial domain contains overlapping, valid Earth observation data.
6. **Step 6: Resampling Provenance Recording**:
   Resampling a native 20m band to a 10m grid via bicubic interpolation produces a $2\times$ pixel array, but does **not** create native 10m high-frequency optical information. Observation metadata explicitly records:
   `native_gsd_m = 20.0`, `grid_gsd_m = 10.0`, `interpolator = 'bicubic'`, `is_upsampled = True`, `alignment_status: PASSED`.

### B.3 Task-Specific Temporal Windows & Operational Cloud Limits
* **Temporal Interval Operational Contracts**:
  * `CHANGE_DETECTION`: Revisit gap $\Delta t = |t_2 - t_1| \ge 5\text{ days}$ (the nominal orbital revisit period of the Sentinel-2 constellation) up to $\le 730\text{ days}$ (2 years). Acquisitions with $\Delta t < 5\text{ days}$ capture intra-day diurnal illumination and transient shadows rather than land-surface alterations. Pairs with $\Delta t > 2\text{ years}$ introduce severe phenological and climatic baseline drift that invalidates bi-temporal change models. These are operational contract bounds tailored to the Sentinel-2 OSCD benchmark.
  * `OPTICAL_SAR_FUSION`: The system inspects query intent to determine dynamic vs static requirements:
    * *Dynamic Tasks (e.g. Flood / Water Mapping)*: Requires $|t_{\text{opt}} - t_{\text{sar}}| \le 72\text{ hours}$ to capture identical hydrological surface conditions.
    * *Static Tasks (e.g. Geological / Topographical Classification)*: Permits $|t_{\text{opt}} - t_{\text{sar}}| \le 10\text{ days}$.
* **Operational Cloud Thresholds**:
  * Optical Land Cover: Cloud/shadow cover $\le 20\%$ of patch area.
  * Optical Change Detection: Cloud/shadow cover $\le 10\%$ in both $T_1$ and $T_2$ images (higher sensitivity due to cloud-edge false positives).
  * SAR: Microwave penetration of clouds; cloud threshold is defined as N/A.
  * These thresholds represent task-specific experimental contract parameters reflecting the benchmark training distributions.

---

## Deliverable C: Routing Baseline Matrix (R0–R5) & Clean Ablation Design

### C.1 2x2 Factorial Ablation Grid for Controlled Factor Isolation
All supervised routing models share the **exact same fine-tuned intent classification backbone**, feature representations, and training split to eliminate confounding factors:

| Model ID | Intent Classifier Backbone | Probability Calibration | Compatibility Gate | Selective Abstention Rule | Factor Isolated in Comparison |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **R0** | Heuristic Keyword Matcher | None | Simulated (Ungated) | None (Argmax dispatch) | External heuristic baseline |
| **R1** | Linear Classifier (TF-IDF + LogReg) | None | Simulated (Ungated) | None (Argmax dispatch) | Standard unconstrained baseline |
| **R1-Gated** | Same Linear Classifier | None | **Active** | None (Argmax dispatch) | **R1 vs R1-Gated**: Isolates Compatibility Gate on linear model |
| **R2** | Fine-Tuned Specialist Classifier | None (Raw Softmax) | **Active** | None (Always dispatches if compatible) | Baseline gated non-selective system |
| **R2-Cal** | Same Fine-Tuned Classifier | **Calibrated** ($T_{\text{cal}}$) | **Active** | None (Always dispatches if compatible) | **R2 vs R2-Cal**: Isolates Calibration alone without abstention |
| **R3** | Same Fine-Tuned Classifier | None (Raw Softmax) | **Active** | **Selective** ($\hat{P}_{\text{raw}} \ge \tau_{\text{sel}}$) | **R2 vs R3**: Isolates Selective Abstention alone on uncalibrated scores |
| **R4** | Same Fine-Tuned Classifier | **Calibrated** ($T_{\text{cal}}$) | **Active** | **Selective** ($P_{\text{cal}} \ge \tau_{\text{sel}}$) | **Proposed System**:<br>• **R3 vs R4**: Isolates Calibration under Selective Abstention<br>• **R2-Cal vs R4**: Isolates Selective Abstention under Calibrated Scores |
| **R5** | Frontier LLM (Few-shot prompt) | Inherent LLM logits | Simulated (Ungated) | Prompted Confidence String | **External LLM Benchmark** (Evaluated separately, not as single-factor ablation) |

### C.2 Separate Evaluation of R5 Frontier LLM Router
* **External Reference Nature**: R5 is evaluated separately as an external zero-shot foundation model baseline, **not** as a single-factor parametric ablation.
* **Non-Calibration of LLMs**: LLM self-reported verbalized confidence strings or raw token log-probabilities are not inherently calibrated (Kadavath et al., 2022). Furthermore, greedy decoding (temperature = 0.0) does not guarantee exact bitwise reproducibility across hosted API updates.
* **Evaluated Metrics for R5**: (1) Task prediction accuracy; (2) Incompatible dispatch proposal rate (IDPR); (3) Unsafe executions if unconstrained vs safe interception under active gate; (4) Clarification behavior on ambiguous queries (hallucinated choice vs clarification prompt); (5) Operational coverage.
* **Frozen API Parameters**: Primary: `gemini-1.5-pro-002`; Secondary cross-evaluation: `gpt-4o-2024-08-06`. Greedy decoding (`temperature = 0.0`, `top_p = 1.0`, `max_output_tokens = 512`, `seed = 42`). November 2026 API release snapshot.

### C.3 Missing-Modality Fallback Governance
* **Prohibition of Silent Substitution**: If user requests multimodal land cover ($T^* = \text{LAND\_COVER\_CLASSIFICATION}$) but attaches only optical or SAR data, the system **MUST NOT** silently dispatch single-sensor models. Doing so violates the non-coercion principle.
* **Deterministic Halt**: System halts with `REJECT_INCOMPATIBLE_OBSERVATION` (`ERR_MISSING_MODALITY`).
* **Segregated Fallback Models**: Fallback specialists (`LandCover_OpticalOnly_ResNet34`, `LandCover_SAROnly_ResNet18`) are registered as distinct specialists with separate task keys, executed only upon explicit user confirmation, and evaluated on segregated single-modality benchmarks.

---

## Deliverable D: Dataset Split & Leakage-Prevention Protocol ($N=700, F=140$)

### D.1 Independent Dual-Annotator Ground-Truth Protocol for Ambiguity ($A^* \in \{0, 1\}$)
To eliminate model-dependent circularity:
* **Definition of Linguistic Ambiguity**: A query $q$ is defined as objectively ambiguous ($A^* = 1$) if the prompt instruction lacks semantic task-discriminating verbs or nouns, permitting two or more plausible interpretations under the 7-task taxonomy $\mathcal{T}$.
* **Annotation Procedure**: Every query in the benchmark is independently evaluated by two remote-sensing domain annotators:
  * Annotators assign $A^* \in \{0, 1\}$ and record plausible intended tasks $\mathcal{T}_{\text{plausible}} \subseteq \mathcal{T}$.
  * If $|\mathcal{T}_{\text{plausible}}| \ge 2$, the query is classified as Ambiguous ($A^* = 1$).
  * If $|\mathcal{T}_{\text{plausible}}| == 1$, the query is classified as Unambiguous ($A^* = 0$).
  * Disagreements are adjudicated by a third expert arbiter. Inter-annotator agreement is reported via Cohen's $\kappa$ (target $\kappa \ge 0.85$).
  * For ambiguous queries ($A^* = 1$), the ground-truth target action is formally defined as $T^* = \text{CLARIFY}$.
* **Zero Model Dependence**: Router confidence scores $c_{\text{intent}}$ are evaluated *against* these human-annotated ground-truth labels.
* **Adversarial Perturbation Audit ($V_4$)**: All $V_4$ queries undergo human review to certify that syntactic distractor keywords (e.g. mentioning radar in a change query) or typographical noise do **not** alter the core task intent.

### D.2 Explicit 1-to-1 Family-to-Cohort Construction Schema
Every template family $f \in \{1, \dots, F\}$ is anchored by a single ground-truth task $T^*(f) \in \mathcal{T}$ and generates **exactly 5 scenario variants** ($V_1, V_2, V_3, V_4, V_5$) mapping 1-to-1 to five mutually exclusive evaluation cohorts:

| Variant ID | Archetype Name | Query Linguistic Transformation | Observation Payload Configuration | Evaluation Function |
| :---: | :--- | :--- | :--- | :--- |
| **$V_1$** | **Canonical Baseline** | Direct, unambiguous task instruction (e.g. "Classify land-cover categories across this scene"). | Fully compliant payload meeting all 8 contract dimensions. | Evaluates baseline specialist routing accuracy. |
| **$V_2$** | **Syntactic Paraphrase** | Paraphrased syntax, regional terminology, passive phrasing ("Identify biophysical surface categories in this observation"). | Fully compliant payload. | Evaluates linguistic generalization without semantic drift. |
| **$V_3$** | **Linguistic Ambiguity** | Human-annotated ambiguous query ($A^* = 1$, ground truth $T^* = \text{CLARIFY}$). | Standard observation payload (compatible with 1, multiple, or 0 tasks). | Verifies mandatory `CLARIFY_AMBIGUOUS_INTENT` (never auto-dispatches). |
| **$V_4$** | **Adversarial Perturbation** | Query contains distracting keywords from competing tasks or typos, verified by human review to preserve $T^*$. | Fully compliant payload for true intent $T^*$. | Evaluates intent extraction robustness against lexical noise. |
| **$V_5$** | **Contract Violation** | Clear intent query ($V_1$ text) paired with a synthetically perturbed payload violating 1 of 8 physical dimensions. | Perturbed payload: missing modality, temporal gap $< 5\text{ d}$, grid-phase shift $> 0.05\text{ px}$, cloud $> 20\%$. | Verifies physical gate halts with `REJECT_INCOMPATIBLE_OBSERVATION` ($\text{IDER} \equiv 0.0\%$). |

**Mutual Exclusivity Guarantee**: No scenario is simultaneously ambiguous and contract-violating. Ambiguity is isolated in $V_3$; payload contract violations are isolated in $V_5$.

### D.3 Benchmark Split Accounting
The benchmark comprises **$N = 700$ multi-task scenarios** organized into **$F = 140$ independent template families** (5 variants per family, 20 families per task across 7 tasks):

| Split | Families ($F$) | Scenarios ($N$) | Families / Task | Scenarios / Task | Template Family Overlap |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Training Split (60%)** | 84 | 420 | 12 | 60 | **0.0%** |
| **Validation Split (20%)** | 28 | 140 | 4 | 20 | **0.0%** |
| **Held-Out Test Split (20%)** | 28 | 140 | 4 | 20 | **0.0%** |
| **Total Benchmark** | **140** | **700** | **20** | **100** | **Strictly Disjoint** |

### D.4 Fixed Allocation Table for Contract Violations ($V_5$) Across Dimensions

| Failure Dimension | Synthetic Anomaly Mechanism | Train ($N=84$) | Val ($N=28$) | Held-Out Test ($N=28$) | Total ($N=140$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Dim 1: Missing Modality** | Omit SAR for Land Cover or Optical for Fusion | 12 | 4 | **4** | 20 |
| **Dim 2: Temporal Deficit** | Single image for Change Detection ($\Delta t = 0$) | 12 | 4 | **4** | 20 |
| **Dim 3: Spectral Deficit** | 3-band RGB supplied for 12-band multispectral task | 12 | 4 | **4** | 20 |
| **Dim 4: Spatial Dimension** | Sub-patch smaller than specialist minimum ($80 \times 80 < 120 \times 120$) | 10 | 3 | **3** | 16 |
| **Dim 5: CRS / Projection** | Unreprojectable CRS or severe reprojection distortion | 10 | 3 | **3** | 16 |
| **Dim 6: GSD / Resolution** | Relative pixel scale mismatch $> 1\%$ without valid upsampling | 10 | 3 | **3** | 16 |
| **Dim 7: Grid-Phase Shift** | Sub-pixel phase offset $\delta_{\text{phase}} > 0.05\text{ px}$ without coregistration | 10 | 3 | **3** | 16 |
| **Dim 8: Corrupt Payload / Cloud** | Cloud fraction exceeds threshold ($>20\%$ LC, $>10\%$ CD) or NaNs | 12 | 4 | **4** | 20 |
| **Total Violations ($V_5$)** | Synthetic anomalies across 8 physical failure dimensions | **84** | **28** | **28** | **140** |

- **Finite-Sample Safety Bound**:
  With $N=28$ independent held-out test violations, observing zero executions ($\text{IDER} \equiv 0.0\%$) yields a one-sided upper 95% Wilson confidence bound of:
  $$p_{\text{upper}} = \frac{3}{28 + 3} \approx 9.68\% \quad (\text{or } 1 - 0.05^{1/28} \approx 10.15\%)$$
  We explicitly document this finite-sample bound in the paper; observing zero failures in 28 tests is an empirical sample guarantee, not a mathematical proof of zero population risk.

### D.5 High-Power Contingency Configuration ($F_{\text{test}} = 42, N_{\text{test}} = 210$)
If validation pilot tuning reveals empirical family standard deviation $\hat{\sigma}_{\text{val}} > 0.065$, the test set symmetrically expands to **$F_{\text{test}} = 42$ families** (6 families per task across 7 tasks; $42 \times 5 = 210$ scenarios):
* **Expansion Rule**: Each task receives exactly 2 additional independent families generated with deterministic seeds (`seed = 1000 + family_idx`).
* **Independence Invariant**: Training ($F_{\text{train}} = 84$) and Validation ($F_{\text{val}} = 28$) splits remain completely unchanged. Family overlap across all splits remains strictly $0.0\%$.

---

## Deliverable E: Metric Definitions, Statistical Plan & Reproducible Power Analysis

### E.1 Calibration Optimization & Validation Grid Search
* Initial nominal thresholds are $\tau_{\text{intent}} = 0.65$ and $\tau_{\text{margin}} = 0.15$.
* Final thresholds are formally optimized via 2D grid search on the 28 validation families over $\tau_{\text{intent}} \in [0.50, 0.85]$ (step 0.05) and $\tau_{\text{margin}} \in [0.05, 0.30]$ (step 0.05), maximizing intent $F_1$ subject to zero false-positive dispatches on validation $V_3$ ambiguous queries.
* Temperature scaling $T_{\text{cal}}$ is fitted strictly on validation queries via negative log-likelihood (NLL) minimization.
* Calibration is evaluated using Top-Label Expected Calibration Error ($\text{ECE}_{\text{top}}$) across $M = 15$ equal-frequency quantile bins with zero-error empty bin handling.

### E.2 Three-Tiered Coverage Formulation
1. **Eligible Population ($\mathcal{E}$)**:
   $$\mathcal{E} = \left\{ i \in \{1, \dots, N_{\text{total}}\} \;\middle|\; c_{\text{intent}, i} \ge \tau_{\text{intent}} \;\land\; \Delta c_i \ge \tau_{\text{margin}} \;\land\; \mathcal{C}(\hat{t}_{\text{intent}, i}, o_i) == 1 \right\}$$
2. **Intent-Confidence Coverage ($\Phi_{\text{intent}}$)**:
   $$\Phi_{\text{intent}}(\tau_{\text{intent}}) = \frac{1}{N_{\text{total}}} \sum_{i=1}^{N_{\text{total}}} \mathbb{I}(c_{\text{intent}, i} \ge \tau_{\text{intent}} \land \Delta c_i \ge \tau_{\text{margin}})$$
3. **Compatibility-Qualified Coverage ($\Phi_{\text{compat}}$)**:
   $$\Phi_{\text{compat}}(\tau_{\text{intent}}) = \frac{|\mathcal{E}|}{N_{\text{total}}}$$
4. **Operational Dispatch Coverage ($\Phi_{\text{dispatch}}$)**:
   $$\tau_{\text{eff}} = \max(\tau_{\text{intent}}, \tau_{\text{sel}})$$
   $$\Phi_{\text{dispatch}}(\tau_{\text{sel}}) = \frac{1}{N_{\text{total}}} \sum_{i=1}^{N_{\text{total}}} \mathbb{I}(c_{\text{intent}, i} \ge \tau_{\text{eff}} \land \Delta c_i \ge \tau_{\text{margin}} \land \mathcal{C}(\hat{t}_{\text{intent}, i}, o_i) == 1)$$

### E.3 Clean Risk-Coverage Curves, Zero-Coverage Convention & Oracle Construction
* **Dispatched Population**: $\mathcal{D}(\tau_{\text{sel}}) = \{i \in \mathcal{E} \mid c_{\text{intent}, i} \ge \tau_{\text{sel}}\}$.
* **Selective Risk**:
  $$\mathcal{R}(\tau_{\text{sel}}) = \frac{1}{|\mathcal{D}(\tau_{\text{sel}})|} \sum_{i \in \mathcal{D}(\tau_{\text{sel}})} \mathbb{I}(\hat{t}_i \ne T_i^*) \quad \text{for } |\mathcal{D}(\tau_{\text{sel}})| > 0$$
* **Empirical Sequence & Tie Handling**:
  Sorting $\mathcal{E}$ by descending confidence yields discrete operating points $(\Phi_1, \mathcal{R}_1), \dots, (\Phi_M, \mathcal{R}_M)$. Instances with identical confidence scores are admitted simultaneously into a single pooled step with pooled average risk.
* **Zero-Coverage Endpoint Convention**:
  Following selective classification standards (Geifman & El-Yaniv, 2017; Ding et al., 2020), no artificial observation is introduced; the empirical curve is evaluated over $[\Phi_1, \Phi_M]$ and continued horizontally to the left:
  $$\mathcal{R}(\Phi) := \mathcal{R}_1 \quad \text{for } 0 \le \Phi \le \Phi_1$$
  reflecting the error rate of the single highest-confidence decision. If no requests qualify ($|\mathcal{E}| = 0$), AURC is reported as $\text{N/A}$ with $\Phi = 0.0\%$.
* **Trapezoidal Integration**:
  $$\text{AURC} = \mathcal{R}_1 \cdot \Phi_1 + \sum_{m=1}^{M-1} \frac{\mathcal{R}_m + \mathcal{R}_{m+1}}{2} (\Phi_{m+1} - \Phi_m)$$
* **The Oracle Population & Excess-AURC**:
  The Oracle evaluates the **exact same eligible population $\mathcal{E}$**. Within $\mathcal{E}$, all correct instances ($\hat{t}_i == T_i^*$) are ranked first (accumulating $\mathcal{R} = 0.0$), followed by incorrect instances. This defines the minimum achievable $\text{AURC}_{\text{oracle}}$:
  $$\text{E-AURC} = \text{AURC} - \text{AURC}_{\text{oracle}}$$

### E.4 Safety Metrics: Proposals vs Executions
* **Incompatible Dispatch Proposal Rate (IDPR)**: Logged at the router stage *prior to the gate*:
  $$\text{IDPR} = \frac{\text{Proposals where } \mathcal{C}(\hat{t}, o) == 0}{\text{Total dispatch proposals}}$$
* **Incompatible Dispatch Execution Rate (IDER)**: Logged at the specialist execution stage *post-gate*:
  $$\text{IDER} \equiv \mathbf{0.0\%} \quad (\text{Guaranteed by mandatory physical gate})$$
* **Per-Dimension Contract Violation Detection Recall**: Reported individually across all 8 failure dimensions:
  $$\text{CVDR}_d = \frac{\text{Detected & intercepted violations in dimension } d}{\text{Total synthetic violations in dimension } d}$$
  with 95% Wilson score confidence intervals. Final safety claims are based strictly on held-out test violations ($N=28$ in default, $N=42$ in contingency).

### E.5 Independently Reproducible Cluster Power Analysis
The simulation script is archived in [`tests/power_analysis_reproducible.py`](file:///C:/Users/B%20NAVANEETH/SatQueryAI/tests/power_analysis_reproducible.py) (fixed seed `42`, 5,000 Monte Carlo runs).

#### Reconciled Bivariate Data-Generating Process (DGP)
Individual system family-level AURCs have standard deviation $\sigma = 0.08$. The difference $\Delta_{k, f} = \text{AURC}_{R4, f} - \text{AURC}_{R_k, f}$ has variance $\sigma_\Delta^2 = 2 \sigma^2 (1 - \rho)$. Each simulation standard deviation maps directly to cross-system correlation:
* $\rho = 0.875 \implies \sigma_\Delta = \sqrt{2 \times 0.08^2 \times (1 - 0.875)} = 0.040$ (Close models, e.g. R3 vs R4).
* $\rho = 0.719 \implies \sigma_\Delta = \sqrt{2 \times 0.08^2 \times (1 - 0.719)} = 0.060$ (Moderate correlation, e.g. R2-Cal vs R4).
* $\rho = 0.500 \implies \sigma_\Delta = \sqrt{2 \times 0.08^2 \times (1 - 0.500)} = 0.080$ (Distant models, e.g. R0 or R1 vs R4).

#### Effect Size Justification
In selective classification benchmarks (Geifman & El-Yaniv, 2017; Ding et al., 2020), baseline unconstrained routers achieve $\text{AURC} \in [0.08, 0.18]$. In Phase 1 and Phase 2 routing benchmarks, the unconstrained baseline exhibited selective risk of $\approx 14\%-18\%$, whereas calibrated compatibility-gated routing reduced risk to $\approx 6\%-8\%$. Targeting minimum detectable effect sizes of $\Delta \in \{0.02, 0.03, 0.04\}$ directly corresponds to a 2 to 4 percentage-point reduction in average risk across the coverage curve.

#### Reproducible Simulation Results (5,000 Iterations, Seed 42)

```
===================================================================================================================
   SATQUERY AI (M3.0) — INDEPENDENTLY REPRODUCIBLE CLUSTER POWER ANALYSIS
   Monte Carlo Iterations: 5,000 | Seed: 42 | Full Holm Step-Down Procedure
===================================================================================================================
F_test  | Delta  | Sigma_D | Cohen d  | Pair rho | 2-sided (0.05)  | Holm Tier-1 (0.01)  | Complete Holm  
-------------------------------------------------------------------------------------------------------------------
24      | 0.02   | 0.040   | 0.50     | 0.875    | 65.4         % | 39.1             % | 58.0         %
24      | 0.02   | 0.060   | 0.33     | 0.719    | 36.0         % | 15.3             % | 29.8         %
24      | 0.02   | 0.080   | 0.25     | 0.500    | 20.5         % | 7.1              % | 16.4         %
24      | 0.03   | 0.040   | 0.75     | 0.875    | 94.4         % | 80.1             % | 91.6         %
24      | 0.03   | 0.060   | 0.50     | 0.719    | 65.0         % | 37.9             % | 57.7         %
24      | 0.03   | 0.080   | 0.38     | 0.500    | 41.2         % | 19.1             % | 35.3         %
24      | 0.04   | 0.040   | 1.00     | 0.875    | 99.6         % | 97.4             % | 99.3         %
24      | 0.04   | 0.060   | 0.67     | 0.719    | 86.9         % | 67.3             % | 82.6         %
24      | 0.04   | 0.080   | 0.50     | 0.500    | 64.8         % | 38.3             % | 58.5         %
-------------------------------------------------------------------------------------------------------------------
28      | 0.02   | 0.040   | 0.50     | 0.875    | 71.2         % | 45.2             % | 66.7         %
28      | 0.02   | 0.060   | 0.33     | 0.719    | 40.0         % | 17.7             % | 35.7         %
28      | 0.02   | 0.080   | 0.25     | 0.500    | 24.6         % | 9.3              % | 21.7         %
28      | 0.03   | 0.040   | 0.75     | 0.875    | 96.8         % | 87.6             % | 95.3         %
28      | 0.03   | 0.060   | 0.50     | 0.719    | 71.9         % | 47.3             % | 67.9         %
28      | 0.03   | 0.080   | 0.38     | 0.500    | 48.9         % | 23.4             % | 43.9         %
28      | 0.04   | 0.040   | 1.00     | 0.875    | 100.0        % | 99.2             % | 99.9         %
28      | 0.04   | 0.060   | 0.67     | 0.719    | 92.2         % | 76.9             % | 89.9         %
28      | 0.04   | 0.080   | 0.50     | 0.500    | 72.0         % | 45.3             % | 67.2         %
-------------------------------------------------------------------------------------------------------------------
42      | 0.02   | 0.040   | 0.50     | 0.875    | 88.1         % | 69.4             % | 87.5         %
42      | 0.02   | 0.060   | 0.33     | 0.719    | 55.9         % | 31.2             % | 54.9         %
42      | 0.02   | 0.080   | 0.25     | 0.500    | 35.2         % | 15.0             % | 33.7         %
42      | 0.03   | 0.040   | 0.75     | 0.875    | 99.7         % | 98.0             % | 99.6         %
42      | 0.03   | 0.060   | 0.50     | 0.719    | 88.5         % | 70.9             % | 87.9         %
42      | 0.03   | 0.080   | 0.38     | 0.500    | 66.2         % | 40.2             % | 64.7         %
42      | 0.04   | 0.040   | 1.00     | 0.875    | 100.0        % | 100.0            % | 100.0        %
42      | 0.04   | 0.060   | 0.67     | 0.719    | 98.6         % | 93.8             % | 98.5         %
42      | 0.04   | 0.080   | 0.50     | 0.500    | 88.4         % | 69.9             % | 87.6         %
-------------------------------------------------------------------------------------------------------------------
56      | 0.02   | 0.040   | 0.50     | 0.875    | 95.0         % | 84.8             % | 94.8         %
56      | 0.02   | 0.060   | 0.33     | 0.719    | 68.4         % | 43.0             % | 68.0         %
56      | 0.02   | 0.080   | 0.25     | 0.500    | 44.5         % | 22.0             % | 44.2         %
56      | 0.03   | 0.040   | 0.75     | 0.875    | 100.0        % | 99.7             % | 100.0        %
56      | 0.03   | 0.060   | 0.50     | 0.719    | 95.6         % | 85.9             % | 95.6         %
56      | 0.03   | 0.080   | 0.38     | 0.500    | 77.9         % | 55.4             % | 77.6         %
56      | 0.04   | 0.040   | 1.00     | 0.875    | 100.0        % | 100.0            % | 100.0        %
56      | 0.04   | 0.060   | 0.67     | 0.719    | 99.9         % | 98.9             % | 99.9         %
56      | 0.04   | 0.080   | 0.50     | 0.500    | 95.5         % | 85.3             % | 95.4         %
-------------------------------------------------------------------------------------------------------------------
```

**Statistical Power Assessment**: Default test split ($F_{\text{test}} = 28$) provides $\ge 92.2\%$ two-sided power for large effect sizes ($d \ge 0.67$). If validation pilot variance reveals $\hat{\sigma}_{\text{val}} > 0.065$, test size expands to the pre-registered contingency configuration $F_{\text{test}} = 42$, providing **$88.5\%$ two-sided power** and **$87.9\%$ complete Holm step-down power** for medium effect sizes ($d=0.50$).

---

## Deliverable F: Deep-Learning Specialist Experiment Specifications

### F.1 Multimodal Land-Cover Specialist (BigEarthNet-14k)
* **Dataset & Verified Tiling**: $N = 13,683$ patches across 4 tiles (`T34TCR`, `T34TEQ`, `T34TFN`, `T34TCS`).
  * Sentinel-1: `(2, 120, 120)`, float32, backscatter in dB `[VH, VV]`. Clipped to $[-35.0, +5.0]\text{ dB}$ (calculated strictly on train split) and linearly scaled via $(\sigma^0 + 35.0) / 40.0$.
  * Sentinel-2: `(10, 120, 120)`, uint16 `[B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]`. Native 20m bands are resampled to 10m grid. Scaled via $\text{DN} / 10,000.0 \in [0.0, 1.0]$.
* **Architecture**: Dual-branch ResNet-34 (SAR branch: 2 input channels; Optical branch: 10 input channels) with cross-attention fusion projecting into 512-dimensional fused representation, feeding 16 sigmoid classification heads.
* **Training Recipe & Hyperparameters**:
  * Optimizer: AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $10^{-2}$).
  * Learning rate: Initial $10^{-4}$ with cosine annealing scheduler down to $10^{-6}$.
  * Batch size: 64. Epochs: 40. Compute budget identical for CB-BCE and ASL.
  * Data Augmentation: Random horizontal/vertical flips, random $90^\circ$ rotations.
  * Early stopping: Patience 7 epochs monitored on validation Macro $F_1$.
  * Seeds: Evaluated over 3 independent random seeds (`42, 1337, 2026`).
  * Checkpoint selection: Highest validation Macro $F_1$ (16 classes).
  * Threshold tuning: Per-class sigmoid threshold sweep $\tau_c \in [0.10, 0.90]$ (step 0.05) on validation split.
* **Loss Configurations**:
  * *Config A: Class-Balanced BCE (CB-BCE)*:
    $$w_{c, +} = \frac{1 - \beta}{1 - \beta^{N_{c, +}}}, \quad w_{c, -} = \frac{1 - \beta}{1 - \beta^{N - N_{c, +}}}, \quad \beta = 0.999$$
    $$\mathcal{L}_{\text{CB-BCE}} = -\frac{1}{16}\sum_{c=1}^{16} \left[ w_{c, +} y_c \log(\sigma(z_c)) + w_{c, -} (1 - y_c) \log(1 - \sigma(z_c)) \right]$$
  * *Config B: Asymmetric Loss (ASL)*:
    $$\mathcal{L}_{\text{ASL}} = -\frac{1}{16}\sum_{c=1}^{16} \left[ y_c (1 - p_c)^{\gamma_+} \log(p_c) + (1 - y_c) (p_{c, m})^{\gamma_-} \log(1 - p_{c, m}) \right]$$
    with $\gamma_+ = 0, \gamma_- = 4, m = 0.05$.
* **Validation & Metrics**: Report Macro $F_1$ (16 classes), Macro $F_1$ (14 test-present classes), Micro $F_1$, and Hamming Loss on held-out test set ($N=3,248$).

### F.2 Bitemporal Change Detection Specialist (OSCD)
* **Dataset & Splits**: 14 training scenes, 4 validation scenes (`montpellier`, `paris`, `rio`, `valencia`), 10 held-out test scenes (`aguasclaras`, `austin`, `beihai`, `berlin`, `brasilia`, `budapest`, `campinas`, `mumbai`, `nantes`, `pisa`).
* **Architectures**:
  * *Primary Model*: Bitemporal Image Transformer (BIT) (Chen et al., 2021). Weight-shared Siamese ResNet-18 backbone (downsampling factor 16); spatial tokenizer extracting $L=64$ semantic tokens per image; 4 transformer encoder layers (intra-temporal context) + 4 transformer decoder layers (cross-temporal differences via cross-attention), total 8 layers; shallow FCN prediction head.
  * *Baseline Model*: Fully Convolutional Siamese Difference (FC-Siam-diff) (Daudt et al., 2018).
* **Training Recipe & Hyperparameters**:
  * Optimizer: AdamW, initial lr $10^{-4}$, weight decay $10^{-4}$, cosine decay to $10^{-6}$.
  * Batch size: 16 (patches $256 \times 256$). Epochs: 50.
  * Loss: $\mathcal{L} = 1.0 \cdot \mathcal{L}_{\text{Focal}}(\gamma=2.0) + 0.5 \cdot \mathcal{L}_{\text{Lovasz-Hinge}}$.
* **Scene-Clustered Uncertainty Estimation (Student's $t$ Jackknife)**:
  Because pixels within an urban scene are spatially autocorrelated, pixels are never treated as independent samples.
  For any metric $\hat{\theta}$ (e.g. Pooled $F_1$ or Macro $F_1$), when test scene $k$ ($k=1, \dots, 10$) is omitted, the metric $\hat{\theta}_{(-k)}$ is recomputed from the remaining 9 test scenes:
  * For Pooled $F_1$: Recompute from accumulated confusion matrix:
    $$\text{TP}_{(-k)} = \sum_{j \ne k} \text{TP}_j, \quad \text{FP}_{(-k)} = \sum_{j \ne k} \text{FP}_j, \quad \text{FN}_{(-k)} = \sum_{j \ne k} \text{FN}_j$$
    $$\hat{\theta}_{(-k)} = \frac{2 \text{TP}_{(-k)}}{2 \text{TP}_{(-k)} + \text{FP}_{(-k)} + \text{FN}_{(-k)}}$$
  * For Macro $F_1$: Recompute arithmetic mean across remaining scenes:
    $$\hat{\theta}_{(-k)} = \frac{1}{K-1} \sum_{j \ne k} F_{1, j}$$
  The leave-one-scene-out jackknife standard error is:
  $$\widehat{\text{SE}}_{\text{jack}} = \sqrt{\frac{K-1}{K} \sum_{k=1}^{K} (\hat{\theta}_{(-k)} - \bar{\theta})^2}$$
  With $K=10$ scenes ($df = K - 1 = 9$), the two-sided critical value is $t_{0.025, 9} = 2.262$:
  $$\text{CI}_{95\%} = \hat{\theta} \pm 2.262 \cdot \widehat{\text{SE}}_{\text{jack}}$$
  Both pooled metrics and individual per-scene metrics across all 10 scenes are reported.
* **Threshold Selection**: Threshold $\tau_{\text{CD}}$ optimized strictly on the 4 validation scenes. **Never tune thresholds on the 10 held-out test scenes.**

### F.3 Specialist Implementation Scope Disclaimer
Phase 3 implements deep-learning perception specialists for **two tasks**: Multimodal Land Cover and Bitemporal Change Detection. The remaining five taxonomy tasks (`SINGLE_IMAGE_VQA`, `CHANGE_VQA`, `VISUAL_GROUNDING`, `SCENE_CAPTIONING`, `OPTICAL_SAR_FUSION`) are evaluated as formal routing, clarification, and safety contract targets; full perception model weights for those five tasks are part of future development phases.

---

## Deliverable G: Git Baseline & Regression Verification Output

### G.1 Git Integrity Verification
```powershell
PS C:\Users\B NAVANEETH\SatQueryAI> git rev-parse "v0.2.0-phase2-freeze^{commit}"
ce7134245f9be6b3876a3a2df1699b2641156f9f

PS C:\Users\B NAVANEETH\SatQueryAI> git rev-parse HEAD
0de5cd4228b897d0f16c60058b057e537ad32dff

PS C:\Users\B NAVANEETH\SatQueryAI> git describe --tags
v0.2.0-phase2-freeze-7-g0de5cd4

PS C:\Users\B NAVANEETH\SatQueryAI> git status --short
(empty - working tree completely clean)
```

### G.2 Test Suite Execution Output (76/76 Passing)
```powershell
PS C:\Users\B NAVANEETH\SatQueryAI> python -m pytest tests/test_m30_clarification_and_routing_invariants.py -v
============================= test session starts =============================
tests/test_m30_clarification_and_routing_invariants.py::test_ambiguous_query_single_compatible_candidate_triggers_clarification PASSED [ 20%]
tests/test_m30_clarification_and_routing_invariants.py::test_ambiguous_query_multiple_compatible_candidates_triggers_clarification PASSED [ 40%]
tests/test_m30_clarification_and_routing_invariants.py::test_ambiguous_query_no_compatible_candidates_triggers_clarification PASSED [ 60%]
tests/test_m30_clarification_and_routing_invariants.py::test_clear_intent_incompatible_observation_rejects_without_coercion PASSED [ 80%]
tests/test_m30_clarification_and_routing_invariants.py::test_clear_intent_compatible_observation_low_selective_confidence_flags_review PASSED [100%]
============================== 5 passed in 9.42s ==============================

PS C:\Users\B NAVANEETH\SatQueryAI> python -m pytest tests/ -q
............................................................................  [100%]
76 passed in 47.23s
```

### G.3 Specific Regression Invariants Verified
1. `test_ambiguous_query_single_compatible_candidate_triggers_clarification`: Confirms that an ambiguous query with exactly one compatible candidate halts with `NEEDS_CLARIFICATION` and never auto-dispatches.
2. `test_ambiguous_query_multiple_compatible_candidates_triggers_clarification`: Confirms multiple compatible candidates halt with `NEEDS_CLARIFICATION`.
3. `test_ambiguous_query_no_compatible_candidates_triggers_clarification`: Confirms zero compatible candidates halt with `NEEDS_CLARIFICATION`.
4. `test_clear_intent_incompatible_observation_rejects_without_coercion`: Confirms unambiguous intent with incompatible payload halts with `INCOMPATIBLE_INPUT` and preserves true intent.
5. `test_clear_intent_compatible_observation_low_selective_confidence_flags_review`: Confirms selective abstention flag triggers when confidence is below selective threshold.

---

## Deliverable H: Novelty Positioning & Related-Work Capability Matrix

### H.1 Detailed Related-Work Capability Comparison Matrix

| System / Framework | Natural Language Task Understanding | Programmatic Sensor & Physical Compatibility Gate | Explicit Task Non-Coercion Guarantee | Mandatory Clarification for Ambiguous Intent | Calibrated Selective Routing | Joint Linguistic + Physical Risk-Coverage Evaluation | Primary Reference / Evidence |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Earth-GPT** (Zhang et al., 2024) | Yes | No | No | No | No | No | End-to-end multimodal VLM; executes on arbitrary image inputs regardless of spectral/temporal compatibility. |
| **GeoChat** (Kuckreja et al., 2024) | Yes | No | No | No | No | No | Grounded RS-VLM; lacks sensor contract checks; coerces ambiguous queries to most likely visual feature. |
| **SkyEye** (Zhan et al., 2024) | Yes | No | No | No | No | No | Visual grounding assistant; lacks temporal interval verification or multi-sensor compatibility gates. |
| **RS-LLaVA** (Li et al., 2024) | Yes | No | No | No | No | No | High-resolution RS assistant; lacks physical payload validation; attempts ungrounded inference on invalid payloads. |
| **Learning to Route (L2R)** (Wang et al., 2021) | Yes | No | Partial | No | No | No | Routes queries across specialized networks; lacks geospatial metadata verification (CRS, GSD, affine phase). |
| **SelectiveNet** (Geifman & El-Yaniv, 2019) | No | No | N/A | No | Yes | No | General selective prediction on closed feature spaces; lacks NLP task routing or physical sensor contracts. |
| **STAC Metadata Validators** (Sankaran et al., 2021) | No | Yes | No | No | No | No | Validates geospatial metadata schemas; lacks natural language intent models; crashes or coerces data on mismatch. |
| **SatQuery AI (Proposed System)** | **Yes** | **Proposed** | **Proposed** | **Proposed** | **Proposed** | **Proposed** | **Subject of experimental evaluation under this protocol.** |

### H.2 The Central Scientific Hypothesis to be Tested
> **Hypothesis**: In Earth observation query routing, routing errors arise from two orthogonally distinct sources of uncertainty: **linguistic ambiguity** in user intent, and **physical incompatibility** in observation payloads. Decoupling linguistic intent estimation from deterministic, machine-enforceable sensor compatibility contracts eliminates incompatible specialist executions ($\text{IDER} \equiv 0.0\%$) and achieves significantly lower selective risk (AURC) than unconstrained routing and zero-shot frontier LLM reasoning.

### H.3 Core Research Objectives of the Study
1. **Formalize Non-Coercive Routing**: Demonstrate that decoupling intent classification from observation compatibility guarantees non-coercion and eliminates incompatible specialist executions ($\text{IDER} \equiv 0.0\%$).
2. **Quantify the Risk-Coverage Benefit**: Evaluate whether calibrated selective prediction with compatibility gating achieves statistically significant reductions in AURC compared to unconstrained, uncalibrated, and non-selective baselines.
3. **Compare Against Foundation Models**: Rigorously benchmark deterministic compatibility routing against frontier LLM zero-shot routing on identical multimodal scenarios.

---

## Deliverable I: Protocol Governance & Scientific Sign-Off Declaration

The following parameters are formally frozen for advisor sign-off:
1. **Benchmark Synthesizer**: Cohort variants generated from deterministic template engine (seed `42`).
2. **Power Contingency**: Default test split ($F_{\text{test}} = 28$); pre-registered expansion to $F_{\text{test}} = 42$ triggered if validation bootstrap variance $\hat{\sigma}_{\text{val}} > 0.065$.
3. **Sign-Off Enforcement**: Milestone M3.0 is **REVISED — AWAITING SCIENTIFIC REVIEW**. No Phase 3 deep-learning model training, benchmark generation, or specialist modifications will commence until formal advisor authorization is received.
