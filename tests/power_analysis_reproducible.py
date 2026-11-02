"""
Reproducible Monte Carlo Power Analysis for M3.0 Experimental Protocol.
SatQuery AI (SIH26167).

Simulates family-clustered paired testing for Delta AURC across:
- Candidate test family sizes F_test in {24, 28, 42, 56}
- Mean effect sizes Delta in {0.02, 0.03, 0.04}
- Standard deviations sigma in {0.04, 0.06, 0.08}
- Both unadjusted two-sided alpha = 0.05 and Holm-Bonferroni adjusted alpha = 0.01 (M=5 comparisons)
"""

import numpy as np
from scipy import stats

def run_power_simulation(
    n_sims=5000,
    seed=42
):
    np.random.seed(seed)
    
    family_sizes = [24, 28, 42, 56]
    deltas = [0.02, 0.03, 0.04]
    sigmas = [0.04, 0.06, 0.08]
    
    print("=" * 105)
    print("   SATQUERY AI (M3.0) — INDEPENDENTLY REPRODUCIBLE CLUSTER POWER ANALYSIS")
    print(f"   Monte Carlo Iterations: {n_sims:,} | Seed: {seed} | Paired Two-Sided Testing")
    print("=" * 105)
    print(f"{'F_test':<7} | {'Delta':<6} | {'Sigma':<6} | {'Cohen d':<8} | {'Power (2-sided 0.05)':<22} | {'Power (Holm-Bonf 0.01)':<24} | {'Power (1-sided 0.05)':<20}")
    print("-" * 105)
    
    results = []
    
    for f in family_sizes:
        for delta in deltas:
            for sigma in sigmas:
                d = delta / sigma
                
                # Vectorized generation: shape (n_sims, f)
                samples = np.random.normal(delta, sigma, size=(n_sims, f))
                
                # Family means, standard errors, and t-statistics
                sample_means = np.mean(samples, axis=1)
                sample_stds = np.std(samples, axis=1, ddof=1)
                se = sample_stds / np.sqrt(f)
                t_stats = sample_means / se
                
                # Two-sided p-values
                p_twosided = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stats), df=f - 1))
                
                # One-sided p-values (superiority)
                p_onesided = 1.0 - stats.t.cdf(t_stats, df=f - 1)
                
                # Power metrics
                power_twosided_05 = np.mean(p_twosided < 0.05)
                power_holm_01 = np.mean(p_twosided < 0.01)  # Most stringent Holm-Bonferroni tier (alpha / 5)
                power_onesided_05 = np.mean(p_onesided < 0.05)
                
                results.append({
                    'f': f,
                    'delta': delta,
                    'sigma': sigma,
                    'd': d,
                    'power_2s_05': power_twosided_05,
                    'power_holm_01': power_holm_01,
                    'power_1s_05': power_onesided_05
                })
                
                print(f"{f:<7} | {delta:<6.2f} | {sigma:<6.2f} | {d:<8.2f} | {power_twosided_05*100:<20.1f}% | {power_holm_01*100:<22.1f}% | {power_onesided_05*100:<18.1f}%")
        print("-" * 105)
        
    print("=" * 105)
    return results

if __name__ == "__main__":
    run_power_simulation()
