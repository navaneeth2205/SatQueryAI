# Milestone M3.2: Deep Bitemporal Change-Detection Specialist Evaluation Audit Report

**SatQuery AI: Academic Machine Learning Research Project**  
**Task**: Bitemporal Change Detection (`CHANGE_DETECTION`)  
**Evaluated Architectures**: 
1. **Fully Convolutional Siamese Difference (FC-Siam-diff)** (Daudt et al., 2018) — Siamese Multi-Scale Encoder + Skip Differencing + Transpose-Conv Decoder. [Best Evaluated Specialist]
2. **Bitemporal Image Transformer (BIT)** (Chen et al., 2021) — Siamese ResNet-18 + Spatial Tokenizer ($L=64$) + 4 Transformer Encoder Layers + 4 Transformer Decoder Layers + FCN Head. [Underperforming Evaluated Architecture]
3. **Classical Baseline**: Phase 2 Normalized Change Vector Analysis (Gaussian-Smoothed CVA).  
**Dataset**: OSCD (Onera Satellite Change Detection, Sentinel-2 Optical RGB)  
**Hardware Platform**: NVIDIA GeForce RTX 2050 (4 GB VRAM)  
**Protocol Version**: Frozen Experimental Protocol `v3.0.0-freeze-rev21`  
**Milestone Status**: **IMPLEMENTATION & EVALUATION COMPLETE; CERTIFICATION TARGET NOT ACHIEVED**

---

## 1. Executive Summary & Scientific Certification Status

Milestone M3.2 delivers the implementation and empirical evaluation of deep bitemporal perception models for Earth Observation change detection under extreme foreground class imbalance (changed pixels comprise $< 3\%$ of total land area across all scenes).

### 1.1 Certification Target Audit: Target Not Met
Under pre-registered protocol `v3.0.0-freeze-rev21` (Section E.2 & F.2 of `PHASE3_M30_PROTOCOL_FREEZE.md`), the formal certification requirement for the deep change-detection specialist is:
$$\text{Macro } F_1^{(\text{CD})} \ge 0.5000 \quad (50.00\%)$$
across the 10 held-out test scenes.

* **Achieved Held-Out Test Macro $F_1$**:
  * **FC-Siam-diff**: $\mathbf{37.66\%}$ ($\text{Macro } F_1 = 0.3766$)
  * **BIT**: $\mathbf{12.95\%}$ ($\text{Macro } F_1 = 0.1295$)
* **Verdict**: **CERTIFICATION TARGET NOT MET**.
* **Protocol Invariance**: The pre-registered certification threshold ($0.50$) is preserved strictly intact without post-hoc relaxation or modification. While milestone implementation, training pipelines, and held-out evaluations are fully complete, the specialist does not qualify as scientifically certified against the frozen $0.50$ target.

### 1.2 Key Empirical Takeaways:
1. **FC-Siam-diff is the Best Evaluated Model**:
   - FC-Siam-diff achieved **$37.66\%$ Test Macro $F_1$** and **$38.18\%$ Pooled $F_1$**, representing a **$+10.35$ percentage point increase in Macro $F_1$** and a **$+4.25$ percentage point increase in Pooled $F_1$** over the Phase 2 Classical Champion (Gaussian-Smoothed CVA at $27.31\%$ Macro $F_1$ / $33.93\%$ Pooled $F_1$).
2. **BIT Underperformed on this Benchmark**:
   - The Bitemporal Image Transformer (BIT) underperformed substantially ($12.95\%$ Macro $F_1$ / $12.60\%$ Pooled $F_1$). It is retained in the evaluation matrix as an evaluated architecture rather than a champion model.
3. **Threshold-Regime Separation**:
   - All primary scorecard metrics are evaluated strictly under validation-tuned thresholds ($\tau^*$ selected on the 4 validation scenes), with fixed-threshold ablations ($\tau = 0.50$) presented in a separately labeled secondary matrix.
4. **Per-Scene Variance (5 of 10 Scenes Met $F_1 \ge 0.50$)**:
   - Under validation-tuned thresholds ($\tau^* = 0.25$), FC-Siam-diff reached $F_1 \ge 0.50$ on 5 of the 10 held-out scenes (`brasilia`: $72.68\%$, `pisa`: $58.09\%$, `budapest`: $54.62\%$, `nantes`: $53.84\%$, and `aguasclaras`: $51.24\%$).
   - On the remaining 5 scenes (`austin`: $14.79\%$, `beihai`: $2.65\%$, `berlin`: $46.71\%$, `campinas`: $10.41\%$, `mumbai`: $11.54\%$), high false-positive rates on sparse foreground targets pulled down the aggregate macro score.

---

## 2. Dataset Partitioning & Leakage Prevention

In the official OSCD benchmark (Daudt et al., 2018), 24 total cities are provided, originally structured as 14 training cities and 10 test cities.

To prevent threshold leakage and support unbiased hyperparameter/checkpoint selection without touching the test set, the 14 training cities were partitioned into **10 training cities** and **4 validation cities**:

| Split | Count | Cities Included | Purpose in Protocol |
| :--- | :---: | :--- | :--- |
| **Train** | 10 | `beirut`, `berrida`, `bordeaux`, `chongqing`, `dubai`, `hong_kong`, `lasvegas`, `milano`, `rennes`, `saclay_w` | Model parameter optimization ($256 \times 256$ patches, stride 64, augmentations) |
| **Validation** | 4 | `montpellier`, `paris`, `rio`, `valencia` | Checkpoint selection and decision threshold sweep ($\tau_{\text{CD}}^*$) |
| **Held-Out Test** | 10 | `aguasclaras`, `austin`, `beihai`, `berlin`, `brasilia`, `budapest`, `campinas`, `mumbai`, `nantes`, `pisa` | Held-out benchmarking unblinded strictly once for final reporting |

*Disjointness Verification*: Verified via [`scripts/verify_oscd_splits.py`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/scripts/verify_oscd_splits.py) and archived in [`artifacts/specialists/oscd_frozen_split_manifest.json`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/artifacts/specialists/oscd_frozen_split_manifest.json). The intersection of all three sets is empty ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$). No city appears in multiple splits.

---

## 3. Authoritative Model Comparison Matrices

All held-out evaluations are performed across the 10 held-out test scenes totaling 3,077,936 pixels.

### 3.1 Primary Comparison Matrix: Validation-Tuned Threshold Regime ($\tau^*$)

Thresholds ($\tau^*$) were selected by sweeping $[0.10, 0.90]$ strictly on the 4 validation scenes to maximize validation Macro $F_1$. **Test scenes were never accessed during threshold selection.**

| Metric | Phase 2 Baseline (Standard CVA) | Phase 2 Baseline (Gaussian CVA) | M3.2 Baseline (FC-Siam-diff) | M3.2 Architecture (BIT) | Best Evaluated Model |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Model Type** | Pixel Radiometric Differencing | Smoothed Radiometric Differencing | Deep Fully Convolutional Siamese | Bitemporal Image Transformer | Deep Feature Learning |
| **Backbone / Depth** | None (Direct RGB) | None (Direct RGB) | 4-Stage ConvNet (1.4M params) | Siamese ResNet-18 + 8-Layer Transformer | FC-Siam-diff |
| **Threshold Regime** | Validation-Tuned ($\tau=0.28$) | Validation-Tuned ($\tau=0.28$) | Validation-Tuned ($\tau^*=\mathbf{0.25}$) | Validation-Tuned ($\tau^*=\mathbf{0.45}$) | Pure Validation Selection |
| **Training Duration** | Instant (CPU) | Instant (CPU) | **5.68 minutes (GPU)** | 6.32 minutes (GPU) | FC-Siam-diff |
| **Best Training Epoch** | — | — | **Epoch 11 / 20** | Epoch 5 / 20 | FC-Siam-diff |
| **Validation Macro $F_1$** | — | — | **23.53%** | 11.43% | FC-Siam-diff |
| **Test Macro $F_1$ (10 Scenes)** | 20.31% | 27.31% | **$\mathbf{37.66\%}$** | 12.95% | **FC-Siam-diff (+10.35% over Classical)** |
| **Test Pooled $F_1$** | 27.78% | 33.93% | **$\mathbf{38.18\%}$** | 12.60% | **FC-Siam-diff (+4.25% over Classical)** |
| **Test Pooled Precision** | **58.20%** | 39.70% | 35.33% | 18.79% | Standard CVA (High Precision) |
| **Test Pooled Recall** | 18.24% | 29.63% | **$\mathbf{41.53\%}$** | 9.47% | **FC-Siam-diff (+11.90% over Classical)** |
| **Test Pooled IoU** | 16.13% | 20.43% | **$\mathbf{23.60\%}$** | 6.72% | **FC-Siam-diff (+3.17% over Classical)** |
| **Certification Status ($\ge 0.50$)**| **FAIL** | **FAIL** | **FAIL (0.3766 < 0.50)** | **FAIL (0.1295 < 0.50)** | Target Not Met |

### 3.2 Secondary Comparison Matrix: Default Fixed Threshold Regime ($\tau = 0.50$)

| Metric | Phase 2 Baseline (Standard CVA) | Phase 2 Baseline (Gaussian CVA) | M3.2 Baseline (FC-Siam-diff) | M3.2 Architecture (BIT) | Best Evaluated Model |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Threshold Regime** | Fixed ($\tau=0.28$) | Fixed ($\tau=0.28$) | Default Fixed ($\tau=0.50$) | Default Fixed ($\tau=0.50$) | Fixed Baseline |
| **Test Macro $F_1$ (10 Scenes)** | 20.31% | 27.31% | **$\mathbf{37.62\%}$** | 10.65% | **FC-Siam-diff** |
| **Test Pooled $F_1$** | 27.78% | 28.18% | **$\mathbf{38.55\%}$** | 8.75% | **FC-Siam-diff** |
| **Test Pooled Precision** | 58.20% | **62.68%** | 39.52% | 22.61% | Gaussian CVA |
| **Test Pooled Recall** | 18.24% | 18.18% | **$\mathbf{37.62\%}$** | 5.43% | **FC-Siam-diff** |
| **Test Pooled IoU** | 16.13% | 16.40% | **$\mathbf{23.88\%}$** | 4.58% | **FC-Siam-diff** |

---

## 4. Granular Per-Scene Breakdown & Metric Reproduction

Evaluated across the 10 held-out scenes using the validation-selected decision threshold ($\tau^* = 0.25$ for FC-Siam-diff; $\tau^* = 0.45$ for BIT), archived in [`artifacts/specialists/m32_per_scene_evaluation.csv`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/artifacts/specialists/m32_per_scene_evaluation.csv):

| Scene | True Change Pixels | Change Ratio | Classical Gaussian CVA $F_1$ | FC-Siam-diff Precision | FC-Siam-diff Recall | FC-Siam-diff $F_1$ | FC-Siam-diff IoU | BIT $F_1$ | Scene Status ($\ge 0.50$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **aguasclaras** | 8,552 | 5.69% | 0.3336 | 0.4018 | 0.7070 | **0.5124** | 0.3445 | 0.3373 | **PASS** |
| **austin** | 5,004 | 1.14% | 0.0298 | 0.0823 | 0.7290 | **0.1479** | 0.0799 | 0.0511 | FAIL |
| **beihai** | 954 | 0.44% | 0.0266 | 0.0140 | 0.2425 | **0.0265** | 0.0134 | 0.0165 | FAIL |
| **berlin** | 28,638 | 7.21% | 0.1259 | 0.4619 | 0.4724 | **0.4671** | 0.3047 | 0.1597 | FAIL |
| **brasilia** | 13,043 | 6.79% | 0.6515 | 0.6221 | 0.8738 | **0.7268** | 0.5708 | 0.1881 | **PASS** |
| **budapest** | 5,233 | 2.58% | 0.4195 | 0.5102 | 0.5878 | **0.5462** | 0.3757 | 0.4118 | **PASS** |
| **campinas** | 1,228 | 1.32% | 0.0607 | 0.0559 | 0.7522 | **0.1041** | 0.0549 | 0.0000 | FAIL |
| **mumbai** | 48,779 | 9.92% | **0.1892** | 0.6216 | 0.0636 | 0.1154 | 0.0612 | 0.0295 | FAIL |
| **nantes** | 2,423 | 0.80% | 0.2185 | 0.6064 | 0.4841 | **0.5384** | 0.3684 | 0.0021 | **PASS** |
| **pisa** | 45,223 | 7.67% | **0.6758** | 0.6812 | 0.5064 | 0.5809 | 0.4094 | 0.0990 | **PASS** |
| **Macro Average** | — | **4.36%** | **27.31%** | **40.45%** | **54.19%** | **$\mathbf{37.66\%}$** | **$\mathbf{21.93\%}$** | **12.95%** | **5 PASS / 5 FAIL** |

### 4.1 Exact Arithmetic Reproduction:
* **FC-Siam-diff Test Macro $F_1$**:
  $$\text{Macro } F_1 = \frac{0.5124 + 0.1479 + 0.0265 + 0.4671 + 0.7268 + 0.5462 + 0.1041 + 0.1154 + 0.5384 + 0.5809}{10} = \frac{3.7657}{10} = \mathbf{0.376570} \quad (37.66\%)$$
* **FC-Siam-diff Confusion Aggregates**:
  * $\text{Total TP} = 66,068, \quad \text{Total FP} = 120,922, \quad \text{Total FN} = 93,009, \quad \text{Total TN} = 2,797,937$
  * $\text{Pooled Precision} = \frac{66068}{66068 + 120922} = \frac{66068}{186990} = \mathbf{0.353324} \quad (35.33\%)$
  * $\text{Pooled Recall} = \frac{66068}{66068 + 93009} = \frac{66068}{159077} = \mathbf{0.415321} \quad (41.53\%)$
  * $\text{Pooled } F_1 = \frac{2 \times 66068}{2 \times 66068 + 120922 + 93009} = \frac{132136}{346067} = \mathbf{0.381822} \quad (38.18\%)$
  * $\text{Pooled IoU} = \frac{66068}{66068 + 120922 + 93009} = \frac{66068}{279999} = \mathbf{0.235958} \quad (23.60\%)$
* **BIT Test Macro $F_1$**:
  $$\text{Macro } F_1 = \frac{0.3373 + 0.0511 + 0.0165 + 0.1597 + 0.1881 + 0.4118 + 0.0000 + 0.0295 + 0.0021 + 0.0990}{10} = \frac{1.2951}{10} = \mathbf{0.129510} \quad (12.95\%)$$
* **BIT Confusion Aggregates**:
  * $\text{Total TP} = 15,071, \quad \text{Total FP} = 65,145, \quad \text{Total FN} = 144,006, \quad \text{Total TN} = 2,853,714$
  * $\text{Pooled } F_1 = \frac{2 \times 15071}{2 \times 15071 + 65145 + 144006} = \frac{30142}{239293} = \mathbf{0.125963} \quad (12.60\%)$

All aggregate scores match the underlying pixel and per-scene arrays to $10^{-6}$ precision.

---

## 5. Architectural Comparison & Scientific Hypotheses

1. **Comparative Superiority of FC-Siam-diff**:
   - In this experimental setup, FC-Siam-diff outperforms BIT by $+24.71$ percentage points in Macro $F_1$ ($37.66\%$ vs. $12.95\%$) and $+25.58$ percentage points in Pooled $F_1$ ($38.18\%$ vs. $12.60\%$).
   - FC-Siam-diff directly computes multi-scale absolute difference maps $|E_i(T_1) - E_i(T_2)|$ at four spatial resolutions ($1\times, 2\times, 4\times, 8\times$), maintaining local geometric continuity across skip connections.

2. **Hypotheses Regarding BIT Underperformance**:
   > [!NOTE] Working Hypothesis
   > The substantial underperformance of BIT on OSCD is hypothesized to stem from data starvation in vision transformer token spaces:
   > - BIT condenses $256 \times 256$ spatial representations into $L=64$ semantic tokens per time step.
   > - In low-data satellite regimes (10 training scenes), the transformer self-attention and cross-attention mechanisms may overfit the specific token distributions of the training cities, failing to reconstruct generalizable difference tokens on unseen test scenes.
   > - This represents an empirical hypothesis consistent with the literature (transformers requiring pretraining on hundreds of thousands of patches before matching convolutional inductive bias on small benchmarks), rather than a demonstrated certainty.

3. **Generalization Limitations**:
   - This experiment reflects a single training run across 10 training scenes and 10 held-out test scenes. While city-level separation guarantees zero spatial overlap, it does not demonstrate broad global or multi-sensor generalization across different biomes or sensor angles.
   - Significant per-scene variance persists: scenes with distinct structural changes (`brasilia`, `pisa`, `budapest`) show high accuracy, whereas scenes with sub-1% change (`beihai`, `austin`, `campinas`) exhibit elevated false positives.

---

## 6. Reproducibility Manifest & Hyperparameter Recipe

To guarantee exact scientific reproducibility, the experimental parameters are archived:
* **Hardware**: NVIDIA GeForce RTX 2050 (4 GB VRAM) on Windows.
* **Batch Size**: 8 patches ($256 \times 256$ pixels).
* **Patch Sampling**: Stride 64 on training scenes (586 patches); random horizontal/vertical flips and 90-degree rotations.
* **Loss Function**: Hybrid BCE with Logits ($\text{pos\_weight}=2.0$) + Dice Loss ($\text{weight}=1.0$).
* **Optimizer**: AdamW ($lr = 5 \times 10^{-4}$ for FC-Siam-diff; $lr = 2 \times 10^{-4}$ for BIT with $0.2\times$ backbone scaling).
* **LR Schedule**: Cosine Annealing with $T_{\text{max}} = 20$, $\eta_{\text{min}} = 10^{-6}$.
* **Checkpoint Selection**: Model with highest validation Macro $F_1$ evaluated at nominal $\tau = 0.50$.
* **Threshold Sweep**: Grid $[0.10, 0.90]$ with step $0.05$ strictly on the 4 validation scenes.
* **Inference Pipeline**: Full-scene overlapping sliding window ($256 \times 256$, stride 128) with Gaussian boundary weighting and reflection padding for sub-256 scenes (`campinas`).

### Key Artifacts on Disk:
* Verification Script: [`scripts/verify_oscd_splits.py`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/scripts/verify_oscd_splits.py)
* Training & Evaluation Script: [`scripts/train_and_evaluate_change_detection.py`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/scripts/train_and_evaluate_change_detection.py)
* Reproduction Script: [`scripts/reproduce_m32_evaluation_report.py`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/scripts/reproduce_m32_evaluation_report.py)
* Comparison Manifest: [`artifacts/specialists/m32_change_detection_comparison.json`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/artifacts/specialists/m32_change_detection_comparison.json)
* Per-Scene CSV: [`artifacts/specialists/m32_per_scene_evaluation.csv`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/artifacts/specialists/m32_per_scene_evaluation.csv)
* Automated Test Suite: [`tests/test_m32_change_detection_evaluation.py`](file:///c:/Users/B%20NAVANEETH/SatQueryAI/tests/test_m32_change_detection_evaluation.py) (2/2 passing).

---

## 7. Recommendations for Milestone M3.3 (Selective Routing)

The failure to meet the $0.50$ certification target across all 10 test scenes does **not** invalidate the research pipeline or require discarding the models. On the contrary, it provides crucial empirical reality for Milestone M3.3:

1. **Selective Router Necessity**:
   - If change-detection models achieved $100\%$ accuracy on all inputs, selective routing would be trivial.
   - The observed per-scene variance (5 scenes passing with $F_1 \in [51\%, 73\%]$ vs. 5 scenes failing with high false positives on sparse targets) demonstrates exactly why the **Input-Constrained Selective Intent Router** must evaluate input sensor contracts, linguistic ambiguity, and confidence margins before specialist dispatch.
2. **Abstention & Clarification Value**:
   - In M3.3, queries directed to change detection on incompatible sensors or ambiguous instructions should be routed to **Pathway 2 (REJECT_INCOMPATIBLE)** or **Pathway 3 (CLARIFY_AMBIGUOUS)**, avoiding silent specialist failure.
   - For qualified requests with low specialist confidence, the system can gracefully exercise **Pathway 1-Alt (ABSTAIN_LOW_CONFIDENCE)**.

**Milestone M3.2 is formally complete with implementation and evaluation validated, and certification target documented as unmet.**
