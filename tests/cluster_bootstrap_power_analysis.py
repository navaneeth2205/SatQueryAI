"""
Fast Paired Cluster-Bootstrap Power Analysis for Global AURC Comparison.
SatQuery AI (SIH26167) - Milestone M3.0 Experimental Protocol.

Simulates:
1. Resampling complete template families as clusters with replacement.
2. Pooling scenarios from resampled families to reconstruct evaluation population.
3. Computing global risk-coverage curve and global AURC for each model.
4. Computing paired difference Delta AURC* = AURC(R_k)* - AURC(R_4)*.
5. Rejection determined via percentile bootstrap confidence intervals and Holm step-down.
"""

import numpy as np

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
    
    # Indices where value changes or at the end
    change_idx = np.where(np.diff(sorted_conf) != 0)[0]
    eval_idx = np.append(change_idx, n - 1)
    
    counts = eval_idx + 1
    coverages = counts / n
    cum_err = np.cumsum(sorted_err)
    risks = cum_err[eval_idx] / counts
    
    # Trapezoidal integration with left continuation R(0) := R_1
    dx = np.diff(coverages)
    trap = 0.5 * (risks[:-1] + risks[1:]) * dx
    aurc = risks[0] * coverages[0] + np.sum(trap)
    return aurc

def run_cluster_bootstrap_power(
    family_sizes=[24, 28, 42, 56],
    delta_aurcs=[0.02, 0.03, 0.04],
    n_sims=300,
    n_boot=250,
    seed=42
):
    np.random.seed(seed)
    scenarios_per_family = 5
    offsets = np.arange(scenarios_per_family)
    
    print("=" * 115, flush=True)
    print("   SATQUERY AI (M3.0) — CLUSTER-BOOTSTRAP GLOBAL AURC STATISTICAL POWER SIMULATION", flush=True)
    print(f"   Monte Carlo Iterations: {n_sims:,} | Bootstrap Replicates per Trial: {n_boot:,} | Seed: {seed}", flush=True)
    print("=" * 115, flush=True)
    print(f"{'F_test':<7} | {'Delta AURC':<11} | {'Bootstrap SE':<13} | {'Two-Sided (0.05)':<18} | {'Holm Tier-1 (0.01)':<20} | {'Assessment':<22}", flush=True)
    print("-" * 115, flush=True)
    
    results = []
    
    for K in family_sizes:
        for delta in delta_aurcs:
            rejections_05 = 0
            rejections_01 = 0
            se_list = []
            
            for sim in range(n_sims):
                # Family random effects modeling intra-cluster correlation
                fam_effects = np.random.normal(0, 0.06, size=K)
                
                # R4: Lower error rate and high confidence alignment
                err_prob_r4 = np.clip(0.08 + np.repeat(fam_effects, scenarios_per_family), 0.01, 0.40)
                err_r4 = (np.random.uniform(size=K * scenarios_per_family) < err_prob_r4).astype(float)
                conf_r4 = np.clip(1.0 - err_prob_r4 + np.random.normal(0, 0.06, size=K * scenarios_per_family), 0.50, 0.99)
                
                # Rk (baseline): Higher error rate corresponding to target Delta AURC
                err_prob_rk = np.clip(0.08 + delta * 2.5 + np.repeat(fam_effects, scenarios_per_family), 0.01, 0.55)
                err_rk = (np.random.uniform(size=K * scenarios_per_family) < err_prob_rk).astype(float)
                conf_rk = np.clip(1.0 - err_prob_rk + np.random.normal(0, 0.14, size=K * scenarios_per_family), 0.50, 0.99)
                
                # Vectorized Cluster Bootstrap index generation: shape (n_boot, K * scenarios_per_family)
                sampled_fams = np.random.choice(K, size=(n_boot, K), replace=True)
                sampled_scens = (sampled_fams[:, :, None] * scenarios_per_family + offsets).reshape(n_boot, -1)
                
                boot_deltas = np.zeros(n_boot)
                for b in range(n_boot):
                    idx = sampled_scens[b]
                    aurc_r4_b = compute_global_aurc_fast(conf_r4[idx], err_r4[idx])
                    aurc_rk_b = compute_global_aurc_fast(conf_rk[idx], err_rk[idx])
                    boot_deltas[b] = aurc_rk_b - aurc_r4_b
                    
                se_list.append(np.std(boot_deltas, ddof=1))
                
                # Two-sided percentile CI for alpha = 0.05
                ci_05_lower = np.percentile(boot_deltas, 2.5)
                ci_05_upper = np.percentile(boot_deltas, 97.5)
                if ci_05_lower > 0 or ci_05_upper < 0:
                    rejections_05 += 1
                    
                # Conservative Holm Tier-1 (alpha = 0.01) CI
                ci_01_lower = np.percentile(boot_deltas, 0.5)
                ci_01_upper = np.percentile(boot_deltas, 99.5)
                if ci_01_lower > 0 or ci_01_upper < 0:
                    rejections_01 += 1
                    
            pwr_05 = rejections_05 / n_sims
            pwr_01 = rejections_01 / n_sims
            mean_se = np.mean(se_list)
            
            assessment = "Adequately powered" if pwr_05 >= 0.80 else "Underpowered (<80%)"
            if pwr_05 >= 0.95:
                assessment = "High power reserve"
                
            results.append({
                "K": K,
                "delta": delta,
                "mean_se": mean_se,
                "pwr_05": pwr_05,
                "pwr_01": pwr_01,
                "assessment": assessment
            })
            print(f"{K:<7} | {delta:<11.2f} | {mean_se:<13.4f} | {pwr_05*100:<16.1f}% | {pwr_01*100:<18.1f}% | {assessment:<22}", flush=True)
        print("-" * 115, flush=True)
        
    print("=" * 115, flush=True)
    return results

if __name__ == "__main__":
    run_cluster_bootstrap_power(family_sizes=[24, 28, 42, 56], delta_aurcs=[0.02, 0.03, 0.04], n_sims=300, n_boot=250)
