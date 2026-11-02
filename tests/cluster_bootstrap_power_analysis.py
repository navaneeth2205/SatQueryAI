"""
Cluster-Bootstrap Power Analysis for Global AURC Comparison in SatQuery AI (SIH26167).
Milestone M3.0 Experimental Protocol — Final Research Specification.

Methodology:
1. Data-Generating Process (DGP):
   - Hierarchical cluster model: F_test template families, each generating 3 valid scenario variants
     (V1 Canonical, V2 Paraphrase, V4 Adversarial), strictly matching the eligible evaluation population E_eval.
   - Cohorts V3 (Ambiguous Intent) and V5 (Contract Violation) are segregated into their dedicated
     evaluations (Clarification PRF1 and Compatibility Recall CVDR/IDER) and excluded from selective routing curves.
   - Intra-cluster correlation modeled via family-level random effect u_f ~ N(0, sigma_fam^2) with sigma_fam = 0.06.
   - Proposed model R4: Base error probability p_4 = clip(0.08 + u_f, 0.01, 0.40);
     calibrated confidence c_4 = clip(1 - p_4 + N(0, 0.06), 0.50, 0.99).
   - Baseline models R_m (m in {0, 1, 1-gated, 2, 3}):
     * R0 (Heuristic): Delta AURC ~ 0.08
     * R1 (Ungated Linear): Delta AURC ~ 0.06
     * R1-Gated (Gated Linear): Delta AURC ~ 0.05
     * R2 (Gated Non-Selective): Delta AURC ~ 0.04
     * R3 (Uncalibrated Selective): Target Delta AURC in {0.02, 0.03, 0.04}
   - Scenario errors e ~ Bernoulli(p), confidences c generated with appropriate dispersion.

2. Statistical Inference:
   - Resampling unit: Entire template families sampled with replacement (Paired Cluster Bootstrap).
   - Scenario pooling: Only the 3 eligible scenarios (V1, V2, V4) belonging to sampled families are pooled,
     guaranteeing exact cohort size N_boot_cohort = 3 * F_test (asserted on every replicate).
   - Global population metric: Empirical risk-coverage curves evaluated over [Phi_1, Phi_M], left-continuation R(0) := R_1,
     and integrated via numerical trapezoid rule to compute global AURC for each system.
   - Paired difference: Delta_m* = AURC(R_m)* - AURC(R_4)*.
   - Comparison-level bootstrap p-value: Estimated using finite-sample continuity correction (Davison & Hinkley, 1997):
     p_m = (sum(Delta_m* <= 0) + 1.0) / (B + 1.0).

3. Multiplicity Control:
   - Full 5-comparison Step-Down Holm Procedure across R4 vs {R0, R1, R1-Gated, R2, R3}:
     Sort p-values: p_(1) <= p_(2) <= p_(3) <= p_(4) <= p_(5).
     Compare against step-down thresholds alpha / (5 - k + 1) for k = 1, ..., 5 (alpha = 0.05).
     Step 1: p_(1) <= 0.010; Step 2: p_(2) <= 0.0125; Step 3: p_(3) <= 0.0167; Step 4: p_(4) <= 0.025; Step 5: p_(5) <= 0.050.
   - Target rejection tracked for R4 vs R3 under Complete Step-Down Holm.
   - Unadjusted test at alpha = 0.05 and conservative first-step (Tier-1) Holm test at alpha / 5 = 0.01 reported.
   - Monte Carlo standard error: SE_MC(p) = sqrt(p * (1 - p) / N_sim).
   - All reported power values are simulation estimates conditional on the stated DGP and sampling variation.
"""

import numpy as np
import time

def compute_global_aurc_fast(conf, err):
    """
    Computes global AURC via vectorized numerical trapezoidal integration.
    Tied confidences pooled into single evaluation steps.
    Left-continuation R(0) := R_1 applied.
    """
    n = len(conf)
    order = np.argsort(-conf)
    sorted_conf = conf[order]
    sorted_err = err[order]
    
    change_idx = np.where(np.diff(sorted_conf) != 0)[0]
    eval_idx = np.append(change_idx, n - 1)
    
    counts = eval_idx + 1
    coverages = counts / n
    cum_err = np.cumsum(sorted_err)
    risks = cum_err[eval_idx] / counts
    
    dx = np.diff(coverages)
    trap = 0.5 * (risks[:-1] + risks[1:]) * dx
    aurc = risks[0] * coverages[0] + np.sum(trap)
    return aurc

def run_cluster_bootstrap_power(
    family_sizes=[28, 42, 56, 84],
    delta_aurcs=[0.02, 0.03, 0.04],
    n_sims=400,
    n_boot=300,
    seed=42
):
    np.random.seed(seed)
    # Strictly 3 valid scenarios (V1, V2, V4) per family belonging to E_eval
    scenarios_per_family = 3
    offsets = np.arange(scenarios_per_family)
    
    print("=" * 132, flush=True)
    print("   SATQUERY AI (M3.0) — CLUSTER-BOOTSTRAP GLOBAL AURC STATISTICAL POWER SIMULATION", flush=True)
    print(f"   Monte Carlo Iterations: {n_sims:,} | Bootstrap Replicates per Trial: {n_boot:,} | Seed: {seed}", flush=True)
    print("   Eligible Population E_eval: Exactly 3 scenarios/family (V1, V2, V4). V3 (clarify) & V5 (violations) segregated.", flush=True)
    print("   Multiplicity: Complete 5-Comparison Step-Down Holm Procedure (R4 vs R0, R1, R1-Gated, R2, R3)", flush=True)
    print("=" * 132, flush=True)
    print(f"{'F_test':<7} | {'Delta':<6} | {'Bootstrap SE':<13} | {'Unadjusted (0.05)':<19} | {'Holm Tier-1 (0.01)':<20} | {'Complete Holm':<18} | {'All-5 Holm':<16} | {'Assessment':<18}", flush=True)
    print("-" * 132, flush=True)
    
    results = []
    
    for K in family_sizes:
        for delta in delta_aurcs:
            t0 = time.time()
            rej_05 = 0
            rej_01 = 0
            rej_holm = 0
            rej_all5 = 0
            se_list = []
            
            for _ in range(n_sims):
                # Family-level cluster random effect
                u = np.random.normal(0, 0.06, size=K)
                
                # R4 (Proposed Calibrated Selective System)
                p4 = np.clip(0.08 + np.repeat(u, scenarios_per_family), 0.01, 0.40)
                e4 = (np.random.uniform(size=K * scenarios_per_family) < p4).astype(float)
                c4 = np.clip(1.0 - p4 + np.random.normal(0, 0.06, size=K * scenarios_per_family), 0.50, 0.99)
                
                # Baseline systems R_m across the 5 comparisons:
                # 0: R0 (Heuristic keyword matcher, Delta ~ 0.08)
                # 1: R1 (Linear ungated classifier, Delta ~ 0.06)
                # 2: R1-Gated (Linear gated classifier, Delta ~ 0.05)
                # 3: R2 (Fine-tuned gated non-selective, Delta ~ 0.04)
                # 4: R3 (Fine-tuned uncalibrated selective, Target Delta in {0.02, 0.03, 0.04})
                baseline_deltas = [0.08, 0.06, 0.05, 0.04, delta]
                p_baselines = [np.clip(0.08 + d * 2.5 + np.repeat(u, scenarios_per_family), 0.01, 0.55) for d in baseline_deltas]
                e_baselines = [(np.random.uniform(size=K * scenarios_per_family) < p).astype(float) for p in p_baselines]
                c_baselines = [np.clip(1.0 - p + np.random.normal(0, 0.12, size=K * scenarios_per_family), 0.50, 0.99) for p in p_baselines]
                
                # Cluster bootstrap: resample whole template families with replacement
                fams = np.random.choice(K, size=(n_boot, K), replace=True)
                scens = (fams[:, :, None] * scenarios_per_family + offsets).reshape(n_boot, -1)
                
                boot_deltas = np.zeros((5, n_boot))
                for b in range(n_boot):
                    idx = scens[b]
                    # Rigorous assertion: ensure bootstrap cohort pools exactly 3 scenarios per family
                    assert len(idx) == K * scenarios_per_family, f"Cohort size mismatch: expected {K * scenarios_per_family}, got {len(idx)}"
                    a4 = compute_global_aurc_fast(c4[idx], e4[idx])
                    for m in range(5):
                        ak = compute_global_aurc_fast(c_baselines[m][idx], e_baselines[m][idx])
                        boot_deltas[m, b] = ak - a4
                        
                # Standard error of primary comparison (R4 vs R3)
                se_list.append(np.std(boot_deltas[4], ddof=1))
                
                # Finite-sample continuity-corrected bootstrap p-values
                p_vals = [(np.sum(boot_deltas[m] <= 0) + 1.0) / (n_boot + 1.0) for m in range(5)]
                
                # Direct unadjusted and tier-1 tests for target comparison (index 4)
                if p_vals[4] < 0.05:
                    rej_05 += 1
                if p_vals[4] < 0.01:
                    rej_01 += 1
                    
                # Full 5-comparison Step-Down Holm procedure
                idx_sorted = np.argsort(p_vals)
                sorted_p = np.array(p_vals)[idx_sorted]
                step_down_thresholds = [0.05 / (5 - step) for step in range(5)]
                rejected_flags = [False] * 5
                for step in range(5):
                    if sorted_p[step] <= step_down_thresholds[step]:
                        rejected_flags[idx_sorted[step]] = True
                    else:
                        break  # Step-down halts at first non-rejection
                        
                if rejected_flags[4]:  # Target comparison R4 vs R3 successfully rejected
                    rej_holm += 1
                if all(rejected_flags):  # All 5 comparisons simultaneously rejected
                    rej_all5 += 1
                    
            p05 = rej_05 / n_sims
            p01 = rej_01 / n_sims
            pholm = rej_holm / n_sims
            pall5 = rej_all5 / n_sims
            se_05 = np.sqrt(p05 * (1 - p05) / n_sims)
            se_01 = np.sqrt(p01 * (1 - p01) / n_sims)
            se_h = np.sqrt(pholm * (1 - pholm) / n_sims)
            se_a = np.sqrt(pall5 * (1 - pall5) / n_sims)
            mean_se = np.mean(se_list)
            
            assessment = "Adequately powered" if pholm >= 0.80 else "Underpowered (<80%)"
            if pholm >= 0.95:
                assessment = "High power reserve"
                
            results.append({
                "K": K,
                "delta": delta,
                "mean_se": mean_se,
                "pwr_05": p05,
                "se_mc_05": se_05,
                "pwr_01": p01,
                "se_mc_01": se_01,
                "pwr_complete_holm": pholm,
                "se_mc_holm": se_h,
                "pwr_all_5": pall5,
                "se_mc_all5": se_a,
                "assessment": assessment
            })
            
            t1 = time.time()
            print(f"{K:<7} | {delta:<6.2f} | {mean_se:<13.4f} | {p05*100:>5.1f}% ± {se_05*100:>4.1f}% | {p01*100:>5.1f}% ± {se_01*100:>4.1f}% | {pholm*100:>5.1f}% ± {se_h*100:>4.1f}% | {pall5*100:>5.1f}% ± {se_a*100:>4.1f}% | {assessment:<18} ({t1-t0:.1f}s)", flush=True)
            
        print("-" * 132, flush=True)
        
    print("=" * 132, flush=True)
    return results

if __name__ == "__main__":
    run_cluster_bootstrap_power(family_sizes=[28, 42, 56, 84], delta_aurcs=[0.02, 0.03, 0.04], n_sims=400, n_boot=300, seed=42)
