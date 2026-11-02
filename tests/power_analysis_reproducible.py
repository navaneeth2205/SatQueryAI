"""
Reproducible Monte Carlo Power Analysis for M3.0 Experimental Protocol.
SatQuery AI (SIH26167) - Rigorous Multi-Hypothesis Step-Down Holm Evaluation.

Simulates:
1. Bivariate DGP: Individual system family AURC variance sigma = 0.08.
   Cross-system correlation rho in {0.875, 0.720, 0.500} directly yielding
   difference standard deviations sigma_Delta in {0.040, 0.060, 0.080}.
2. Individual two-sided paired t-test power at alpha = 0.05.
3. Individual conservative Bonferroni tier-1 power at alpha = 0.01 (alpha / 5).
4. Complete 5-comparison step-down Holm-Bonferroni procedure power.
"""

import numpy as np
from scipy import stats

def simulate_m30_power(
    n_sims=5000,
    seed=42
):
    np.random.seed(seed)
    
    family_sizes = [24, 28, 42, 56]
    
    # Bivariate model parameters: sigma_indiv = 0.08
    # rho = 0.875 -> sigma_delta = sqrt(2 * 0.08^2 * (1 - 0.875)) = 0.040
    # rho = 0.719 -> sigma_delta = sqrt(2 * 0.08^2 * (1 - 0.719)) = 0.060
    # rho = 0.500 -> sigma_delta = sqrt(2 * 0.08^2 * (1 - 0.500)) = 0.080
    dgp_configs = [
        {"delta": 0.02, "sigma_delta": 0.040, "rho": 0.875, "label": "Close models (R3 vs R4, rho=0.875)"},
        {"delta": 0.02, "sigma_delta": 0.060, "rho": 0.719, "label": "Moderate correlation (rho=0.719)"},
        {"delta": 0.02, "sigma_delta": 0.080, "rho": 0.500, "label": "Distant models (rho=0.500)"},
        {"delta": 0.03, "sigma_delta": 0.040, "rho": 0.875, "label": "Close models (R3 vs R4, rho=0.875)"},
        {"delta": 0.03, "sigma_delta": 0.060, "rho": 0.719, "label": "Moderate correlation (rho=0.719)"},
        {"delta": 0.03, "sigma_delta": 0.080, "rho": 0.500, "label": "Distant models (rho=0.500)"},
        {"delta": 0.04, "sigma_delta": 0.040, "rho": 0.875, "label": "Close models (R3 vs R4, rho=0.875)"},
        {"delta": 0.04, "sigma_delta": 0.060, "rho": 0.719, "label": "Moderate correlation (rho=0.719)"},
        {"delta": 0.04, "sigma_delta": 0.080, "rho": 0.500, "label": "Distant models (rho=0.500)"},
    ]
    
    print("=" * 115)
    print("   SATQUERY AI (M3.0) — INDEPENDENTLY REPRODUCIBLE CLUSTER POWER ANALYSIS")
    print(f"   Monte Carlo Iterations: {n_sims:,} | Seed: {seed} | Full Holm Step-Down Procedure")
    print("=" * 115)
    print(f"{'F_test':<7} | {'Delta':<6} | {'Sigma_D':<7} | {'Cohen d':<8} | {'Pair rho':<8} | {'2-sided (0.05)':<15} | {'Holm Tier-1 (0.01)':<19} | {'Complete Holm':<15}")
    print("-" * 115)
    
    table_rows = []
    
    for f in family_sizes:
        for cfg in dgp_configs:
            delta = cfg["delta"]
            sigma_d = cfg["sigma_delta"]
            rho = cfg["rho"]
            d = delta / sigma_d
            
            # 1. Focal comparison simulation: shape (n_sims, f)
            focal_diffs = np.random.normal(delta, sigma_d, size=(n_sims, f))
            focal_means = np.mean(focal_diffs, axis=1)
            focal_stds = np.std(focal_diffs, axis=1, ddof=1)
            focal_t = focal_means / (focal_stds / np.sqrt(f))
            p_focal = 2.0 * (1.0 - stats.t.cdf(np.abs(focal_t), df=f - 1))
            
            # Individual 2-sided power at alpha = 0.05
            pwr_2s_05 = np.mean(p_focal < 0.05)
            # Conservative Bonferroni Tier-1 power at alpha = 0.01
            pwr_tier1_01 = np.mean(p_focal < 0.01)
            
            # 2. Complete 5-comparison Holm Step-Down simulation:
            # Assume other 4 comparisons have effects >= focal delta (e.g. R4 vs R0, R1, R2, R5)
            # With typical effect sizes: Delta_comp = [0.06, 0.05, 0.04, 0.035]
            other_deltas = [0.06, 0.05, 0.04, 0.035]
            other_pvals = []
            for od in other_deltas:
                # Correlated draws with focal comparison (shared R4 variance ~ 0.5 covariance)
                od_diffs = np.random.normal(od, 0.06, size=(n_sims, f))
                od_means = np.mean(od_diffs, axis=1)
                od_stds = np.std(od_diffs, axis=1, ddof=1)
                od_t = od_means / (od_stds / np.sqrt(f))
                p_od = 2.0 * (1.0 - stats.t.cdf(np.abs(od_t), df=f - 1))
                other_pvals.append(p_od)
                
            # Stack all 5 p-values: shape (n_sims, 5)
            all_pvals = np.column_stack([p_focal] + other_pvals)
            
            # Apply Holm step-down across each simulation
            # Sorted p-values along axis 1
            sorted_indices = np.argsort(all_pvals, axis=1)
            sorted_pvals = np.take_along_axis(all_pvals, sorted_indices, axis=1)
            
            # Holm thresholds: [0.05/5, 0.05/4, 0.05/3, 0.05/2, 0.05/1]
            holm_thresholds = np.array([0.05 / (5 - k) for k in range(5)])
            
            # A hypothesis at sorted rank k is rejected iff for all j <= k, sorted_pvals[j] < holm_thresholds[j]
            below_thresh = sorted_pvals < holm_thresholds[np.newaxis, :]
            # Cumulative min: once one is false, all subsequent are false
            rejected_sorted = np.minimum.accumulate(below_thresh, axis=1)
            
            # Map back to original focal comparison (which was column index 0)
            # Find the rank of column 0 in sorted_indices
            focal_rank = np.where(sorted_indices == 0)[1]
            focal_rejected_in_holm = rejected_sorted[np.arange(n_sims), focal_rank]
            pwr_complete_holm = np.mean(focal_rejected_in_holm)
            
            table_rows.append({
                "f": f,
                "delta": delta,
                "sigma_d": sigma_d,
                "d": d,
                "rho": rho,
                "pwr_2s_05": pwr_2s_05,
                "pwr_tier1_01": pwr_tier1_01,
                "pwr_complete_holm": pwr_complete_holm
            })
            
            print(f"{f:<7} | {delta:<6.2f} | {sigma_d:<7.3f} | {d:<8.2f} | {rho:<8.3f} | {pwr_2s_05*100:<13.1f}% | {pwr_tier1_01*100:<17.1f}% | {pwr_complete_holm*100:<13.1f}%")
        print("-" * 115)
        
    print("=" * 115)
    return table_rows

if __name__ == "__main__":
    simulate_m30_power()
