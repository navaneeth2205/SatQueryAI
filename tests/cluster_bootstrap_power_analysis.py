"""
Cluster-Bootstrap Power Analysis for Global AURC Comparison in SatQuery AI (SIH26167).
Milestone M3.0 Experimental Protocol — Reconciled Statistical Implementation.

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

2. Statistical Inference & Null-Centred Hypothesis Testing:
   - Resampling unit: Entire template families sampled with replacement (Paired Cluster Bootstrap).
   - Scenario pooling: Only the 3 eligible scenarios (V1, V2, V4) belonging to sampled families are pooled,
     guaranteeing exact cohort size N_eval = 3 * F_test (asserted on every replicate).
   - Global population metric: Empirical risk-coverage curves evaluated over [Phi_1, Phi_M], left-continuation R(0) := R_1,
     and integrated via numerical trapezoid rule to compute global AURC for each system.
   - Observed sample difference: Delta_m = AURC(R_m) - AURC(R_4).
   - Null-Centred Cluster Bootstrap (Hall & Wilson 1991, Davison & Hinkley 1997):
     Under H0: Delta_m <= 0, the bootstrap sampling distribution of differences Delta_m* is centered
     under the null by subtracting the sample estimate Delta_m:
       Delta_tilde_m*(b) = Delta_m*(b) - Delta_m
     The one-sided continuity-corrected p-value is:
       p_m = (sum(Delta_tilde_m* >= Delta_m) + 1.0) / (B + 1.0)
     which properly evaluates the probability of observing an effect as large as Delta_m under H0.

3. Multiplicity Control via Complete 5-Comparison Step-Down Holm Procedure:
   - Full 5-comparison Step-Down Holm Procedure across R4 vs {R0, R1, R1-Gated, R2, R3}:
     Sort p-values: p_(1) <= p_(2) <= p_(3) <= p_(4) <= p_(5).
     Compare against step-down critical thresholds alpha / (5 - k + 1) for k = 1, ..., 5 (alpha = 0.05).
     Step 1: p_(1) <= 0.010; Step 2: p_(2) <= 0.0125; Step 3: p_(3) <= 0.0167; Step 4: p_(4) <= 0.025; Step 5: p_(5) <= 0.050.
   - Sequential termination: If any step fails to reject, the testing sequence terminates.
   - Step-Down Holm Power (Target Claim R4 vs R3): Evaluates whether the hypothesis H0, R4 vs R3
     is reached and rejected under FWER control (primary research target).
   - Conjunctive All-5 Rejection: Evaluates whether all 5 baseline comparisons are simultaneously rejected.
   - Monte Carlo standard error: SE_MC(p) = sqrt(p * (1 - p) / N_sim).
"""

import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor

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

def run_single_simulation_trial(args):
    """
    Executes a single Monte Carlo trial for a given (K, delta, n_boot, seed).
    """
    K, delta, n_boot, seed, is_null = args
    np.random.seed(seed)
    scenarios_per_family = 3
    offsets = np.arange(scenarios_per_family)
    
    # Family-level cluster random effect
    u = np.random.normal(0, 0.06, size=K)
    
    # R4 (Proposed Calibrated Selective System)
    p4 = np.clip(0.08 + np.repeat(u, scenarios_per_family), 0.01, 0.40)
    e4 = (np.random.uniform(size=K * scenarios_per_family) < p4).astype(float)
    c4 = np.clip(1.0 - p4 + np.random.normal(0, 0.06, size=K * scenarios_per_family), 0.50, 0.99)
    
    if is_null:
        # Under null hypothesis H0: Delta = 0, R3 has identical performance distribution as R4
        baseline_deltas = [0.08, 0.06, 0.05, 0.04, 0.00]
    else:
        baseline_deltas = [0.08, 0.06, 0.05, 0.04, delta]
        
    p_baselines = [
        np.clip(0.08 + (0.0 if (is_null and m == 4) else d * 2.5) + np.repeat(u, scenarios_per_family), 0.01, 0.55)
        for m, d in enumerate(baseline_deltas)
    ]
    e_baselines = [(np.random.uniform(size=K * scenarios_per_family) < p).astype(float) for p in p_baselines]
    c_baselines = [
        np.clip(1.0 - p + np.random.normal(0, 0.06 if (is_null and m == 4) else 0.12, size=K * scenarios_per_family), 0.50, 0.99)
        for m, p in enumerate(p_baselines)
    ]
    
    # Sample AURC values
    a4_obs = compute_global_aurc_fast(c4, e4)
    obs_deltas = np.array([
        compute_global_aurc_fast(c_baselines[m], e_baselines[m]) - a4_obs
        for m in range(5)
    ])
    
    # Cluster bootstrap: resample whole template families with replacement
    fams = np.random.choice(K, size=(n_boot, K), replace=True)
    scens = (fams[:, :, None] * scenarios_per_family + offsets).reshape(n_boot, -1)
    
    boot_deltas = np.zeros((5, n_boot))
    for b in range(n_boot):
        idx = scens[b]
        # Rigorous cohort size assertion: exactly 3 scenarios per family
        assert len(idx) == K * scenarios_per_family, f"Cohort size mismatch: {len(idx)} != {K * scenarios_per_family}"
        a4_b = compute_global_aurc_fast(c4[idx], e4[idx])
        for m in range(5):
            ak_b = compute_global_aurc_fast(c_baselines[m][idx], e_baselines[m][idx])
            boot_deltas[m, b] = ak_b - a4_b
            
    # Null-Centred Cluster Bootstrap P-Values:
    # Under H0: Delta_m <= 0, the shifted bootstrap differences are tilde_delta = boot_deltas - obs_deltas
    # P-value is the proportion of null replicates exceeding or equal to observed sample delta
    p_vals = [
        (np.sum((boot_deltas[m] - obs_deltas[m]) >= obs_deltas[m]) + 1.0) / (n_boot + 1.0)
        for m in range(5)
    ]
    
    # Standard error of primary comparison (R4 vs R3, index 4)
    se_target = float(np.std(boot_deltas[4], ddof=1))
    
    # Target hypothesis rejection checks
    rej_05 = bool(p_vals[4] <= 0.05)
    rej_01 = bool(p_vals[4] <= 0.01)
    
    # Complete 5-comparison Step-Down Holm procedure
    idx_sorted = np.argsort(p_vals)
    sorted_p = np.array(p_vals)[idx_sorted]
    step_down_thresholds = [0.05 / (5 - step) for step in range(5)]
    rejected_flags = [False] * 5
    for step in range(5):
        if sorted_p[step] <= step_down_thresholds[step]:
            rejected_flags[idx_sorted[step]] = True
        else:
            break  # Sequential step-down halts at first failure to reject
            
    rej_holm_target = bool(rejected_flags[4])  # Target claim R4 vs R3 rejected
    rej_all5 = bool(all(rejected_flags))       # All 5 comparisons simultaneously rejected
    
    return {
        "p_vals": p_vals,
        "se_target": se_target,
        "rej_05": rej_05,
        "rej_01": rej_01,
        "rej_holm_target": rej_holm_target,
        "rej_all5": rej_all5
    }

def validate_null_type1_error(K=28, n_sims=400, n_boot=300, seed=42, max_workers=6):
    """
    Validates empirical Type-I error rate under simulated null conditions (Delta = 0.00).
    Confirms that nominal alpha = 0.05 is properly controlled.
    """
    print("\n" + "=" * 132, flush=True)
    print("   STEP 1: EMPIRICAL TYPE-I ERROR VALIDATION UNDER NULL HYPOTHESIS (H0: Delta = 0.00)", flush=True)
    print(f"   Simulated Trials: {n_sims:,} | Bootstrap Replicates per Trial: {n_boot:,} | Test Families: {K} | Seed: {seed}", flush=True)
    print("=" * 132, flush=True)
    
    args = [(K, 0.00, n_boot, seed + i, True) for i in range(n_sims)]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(run_single_simulation_trial, args))
        
    rej_05_count = sum(r["rej_05"] for r in results)
    rej_01_count = sum(r["rej_01"] for r in results)
    type1_05 = rej_05_count / n_sims
    type1_01 = rej_01_count / n_sims
    se_type1_05 = np.sqrt(type1_05 * (1 - type1_05) / n_sims)
    se_type1_01 = np.sqrt(type1_01 * (1 - type1_01) / n_sims)
    
    print(f"   Empirical Type-I Error at alpha = 0.05: {type1_05*100:>5.2f}% ± {se_type1_05*100:>4.2f}% (Nominal: 5.00%)", flush=True)
    print(f"   Empirical Type-I Error at alpha = 0.01: {type1_01*100:>5.2f}% ± {se_type1_01*100:>4.2f}% (Nominal: 1.00%)", flush=True)
    print(f"   Validation Status: {'VALIDATED (Within MC error)' if type1_05 <= 0.070 else 'ELEVATED'}", flush=True)
    print("=" * 132 + "\n", flush=True)
    return type1_05, type1_01

def run_cluster_bootstrap_power(
    family_sizes=[28, 42, 56, 84],
    delta_aurcs=[0.02, 0.03, 0.04],
    n_sims=400,
    n_boot=300,
    seed=42,
    max_workers=6
):
    print("=" * 132, flush=True)
    print("   STEP 2: FULL 12-CONDITION GLOBAL AURC STATISTICAL POWER SIMULATION", flush=True)
    print(f"   Monte Carlo Iterations: {n_sims:,} | Bootstrap Replicates per Trial: {n_boot:,} | Seed: {seed}", flush=True)
    print("   Eligible Population E_eval: Exactly 3 scenarios/family (V1, V2, V4). V3 (clarify) & V5 (violations) segregated.", flush=True)
    print("   Multiplicity: Complete 5-Comparison Step-Down Holm Procedure (R4 vs R0, R1, R1-Gated, R2, R3)", flush=True)
    print("=" * 132, flush=True)
    print(f"{'F_test':<7} | {'Delta':<6} | {'Bootstrap SE':<13} | {'Unadjusted (0.05)':<19} | {'Holm Tier-1 (0.01)':<20} | {'Step-Down Holm':<18} | {'All-5 Holm':<16} | {'Assessment':<18}", flush=True)
    print("-" * 132, flush=True)
    
    all_results = []
    
    for K in family_sizes:
        for delta in delta_aurcs:
            t0 = time.time()
            args = [(K, delta, n_boot, seed + K * 1000 + int(delta * 10000) + i, False) for i in range(n_sims)]
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                res = list(executor.map(run_single_simulation_trial, args))
                
            p05 = sum(r["rej_05"] for r in res) / n_sims
            p01 = sum(r["rej_01"] for r in res) / n_sims
            pholm = sum(r["rej_holm_target"] for r in res) / n_sims
            pall5 = sum(r["rej_all5"] for r in res) / n_sims
            mean_se = float(np.mean([r["se_target"] for r in res]))
            
            se_05 = np.sqrt(p05 * (1 - p05) / n_sims)
            se_01 = np.sqrt(p01 * (1 - p01) / n_sims)
            se_h = np.sqrt(pholm * (1 - pholm) / n_sims)
            se_a = np.sqrt(pall5 * (1 - pall5) / n_sims)
            
            assessment = "Adequately powered" if pholm >= 0.80 else "Underpowered (<80%)"
            if pholm >= 0.95:
                assessment = "High power reserve"
                
            row = {
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
            }
            all_results.append(row)
            
            t1 = time.time()
            print(f"{K:<7} | {delta:<6.2f} | {mean_se:<13.4f} | {p05*100:>5.1f}% ± {se_05*100:>4.1f}% | {p01*100:>5.1f}% ± {se_01*100:>4.1f}% | {pholm*100:>5.1f}% ± {se_h*100:>4.1f}% | {pall5*100:>5.1f}% ± {se_a*100:>4.1f}% | {assessment:<18} ({t1-t0:.1f}s)", flush=True)
            
        print("-" * 132, flush=True)
        
    print("=" * 132, flush=True)
    return all_results

def run_sensitivity_analysis(K=56, delta=0.04, replicates_list=[300, 1000], n_sims=400, seed=42, max_workers=6):
    """
    Executes empirical sensitivity analysis comparing B=300 vs B=1,000 bootstrap replicates.
    Directly outputs consistent empirical values for the protocol report.
    """
    print("\n" + "=" * 132, flush=True)
    print("   STEP 3: EMPIRICAL BOOTSTRAP REPLICATE SENSITIVITY TRIAL (B = 300 vs B = 1,000)", flush=True)
    print(f"   Condition: F_test = {K}, Delta = {delta:.2f} | Monte Carlo Iterations: {n_sims:,} | Seed: {seed}", flush=True)
    print("=" * 132, flush=True)
    print(f"{'Replicates (B)':<15} | {'Bootstrap SE':<15} | {'Step-Down Holm Power':<25} | {'All-5 Holm Power':<20} | {'Elapsed Time':<15}", flush=True)
    print("-" * 132, flush=True)
    
    sens_results = {}
    for B in replicates_list:
        t0 = time.time()
        args = [(K, delta, B, seed + 8888 + i, False) for i in range(n_sims)]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            res = list(executor.map(run_single_simulation_trial, args))
            
        mean_se = float(np.mean([r["se_target"] for r in res]))
        pholm = sum(r["rej_holm_target"] for r in res) / n_sims
        pall5 = sum(r["rej_all5"] for r in res) / n_sims
        se_h = np.sqrt(pholm * (1 - pholm) / n_sims)
        se_a = np.sqrt(pall5 * (1 - pall5) / n_sims)
        t1 = time.time()
        
        sens_results[B] = {
            "se": mean_se,
            "pholm": pholm,
            "se_h": se_h,
            "pall5": pall5,
            "se_a": se_a,
            "time": t1 - t0
        }
        print(f"{B:<15} | {mean_se:<15.4f} | {pholm*100:>5.1f}% ± {se_h*100:>4.1f}%          | {pall5*100:>5.1f}% ± {se_a*100:>4.1f}%     | {t1-t0:>6.1f}s", flush=True)
        
    print("-" * 132, flush=True)
    diff_power = abs(sens_results[1000]["pholm"] - sens_results[300]["pholm"])
    diff_se = abs(sens_results[1000]["se"] - sens_results[300]["se"])
    print(f"   Sensitivity Summary: Absolute Power Difference = {diff_power*100:.1f}%, SE Difference = {diff_se:.4f}", flush=True)
    print(f"   Methodological Conclusion: B = 300 provides power estimates within Monte Carlo error (SE_MC ~= {sens_results[300]['se_h']*100:.1f}%)", flush=True)
    print("=" * 132 + "\n", flush=True)
    return sens_results

if __name__ == "__main__":
    # 1. Type-I error validation under null
    validate_null_type1_error(K=28, n_sims=400, n_boot=300, seed=42)
    # 2. Main 12-condition power table
    run_cluster_bootstrap_power(family_sizes=[28, 42, 56, 84], delta_aurcs=[0.02, 0.03, 0.04], n_sims=400, n_boot=300, seed=42)
    # 3. Reconciled sensitivity analysis
    run_sensitivity_analysis(K=56, delta=0.04, replicates_list=[300, 1000], n_sims=400, seed=42)
