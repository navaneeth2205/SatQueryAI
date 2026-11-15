import pandas as pd
import numpy as np
import scipy.stats as stats
import json

df_audit = pd.read_csv('artifacts/benchmarks/m33_scenario_level_audit.csv')
print(f"Loaded {len(df_audit)} scenarios from audit CSV.")

def get_success_indicator(row, model_id):
    c = row['cohort']
    action = row[f'{model_id}_action']
    is_correct = row[f'{model_id}_correct_intent']
    if c == '1_supported_compatible':
        return 1 if (action == 'DISPATCH_QUALIFIED' and is_correct) else 0
    elif c in ('2_supported_incompatible_input', '3_missing_sensor_modalities'):
        return 1 if action == 'REJECT_INCOMPATIBLE' else 0
    elif c == '4_ambiguous_multi_intent':
        return 1 if action in ('CLARIFY_AMBIGUOUS', 'HALTED_UNCERTAINTY') else 0
    elif c in ('5_unsupported_tasks', '6_out_of_distribution'):
        return 1 if action in ('REFUSED_UNAVAILABLE', 'ABSTAIN_LOW_CONFIDENCE', 'CLARIFY_AMBIGUOUS', 'REJECT_INCOMPATIBLE') else 0
    return 0

for m in ['R0', 'R1', 'R1-Gated', 'R2', 'R3', 'R4']:
    df_audit[f'{m}_succ'] = df_audit.apply(lambda r: get_success_indicator(r, m), axis=1)
    print(f"{m} total successes: {df_audit[f'{m}_succ'].sum()} / {len(df_audit)}")

print("\n" + "="*70)
r4_succ = df_audit['R4_succ'].values
comparisons = ['R1', 'R0', 'R2', 'R1-Gated', 'R3']

results = {}
for b in comparisons:
    b_succ = df_audit[f'{b}_succ'].values
    n00 = int(np.sum((r4_succ == 0) & (b_succ == 0)))
    n01 = int(np.sum((r4_succ == 0) & (b_succ == 1)))
    n10 = int(np.sum((r4_succ == 1) & (b_succ == 0)))
    n11 = int(np.sum((r4_succ == 1) & (b_succ == 1)))
    
    b_c = n01 + n10
    stat = ((abs(n10 - n01) - 1.0) ** 2) / float(b_c)
    raw_p = float(stats.chi2.sf(stat, df=1))
    
    results[b] = {
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "n01_plus_n10": b_c,
        "numerator": (abs(n10 - n01) - 1.0) ** 2,
        "chi2": stat,
        "raw_p": raw_p
    }
    print(f"R4 vs {b}:")
    print(f"  Contingency [n00, n01, n10, n11]: [{n00}, {n01}, {n10}, {n11}]")
    print(f"  Sum of counts: {n00 + n01 + n10 + n11}")
    print(f"  R4 success (n10 + n11): {n10 + n11}, Base success (n01 + n11): {n01 + n11}")
    print(f"  Discordant: n01={n01}, n10={n10}, sum={b_c}, diff={abs(n10 - n01)}")
    print(f"  Formula: (|{n10} - {n01}| - 1)^2 / ({n10} + {n01}) = ({abs(n10 - n01) - 1.0})^2 / {b_c} = {(abs(n10 - n01) - 1.0)**2:.1f} / {b_c} = {stat:.4f}")
    print(f"  McNemar Chi2: {stat:.4f}")
    print(f"  Raw p-value: {raw_p:.8e}")
    print()

# Now Holm adjustment
sorted_comps = sorted(comparisons, key=lambda b: results[b]["raw_p"])
M = len(sorted_comps)
cum_max = 0.0
print("="*70)
print("Step-Down Holm Adjustment:")
for rank_idx, b in enumerate(sorted_comps):
    k = rank_idx + 1
    mult = M - k + 1
    raw_p = results[b]["raw_p"]
    adj_p = min(1.0, max(cum_max, raw_p * mult))
    cum_max = adj_p
    reject = adj_p < 0.05
    print(f"Rank {k}: R4 vs {b}")
    print(f"  Raw p: {raw_p:.8e}, Multiplier: {mult}, Holm p: {adj_p:.8e}, Reject H0: {reject}")
