import pandas as pd
import numpy as np

df = pd.read_csv(r'c:\Users\B NAVANEETH\SatQueryAI\artifacts\phase1_router\phase1_locked_validation_predictions.csv')

tau = 0.65

# Reviews: conf < tau
b_rev = df['cand0_baseline_word_lr_t1_conf'] < tau
c_rev = df['cand1_char_wb_lr_t1_conf'] < tau

print(f'Total validation queries: {len(df)}')
print(f'Baseline reviews (< 0.65): {b_rev.sum()} / {len(df)} ({b_rev.mean():.4f})')
print(f'Champion reviews (< 0.65): {c_rev.sum()} / {len(df)} ({c_rev.mean():.4f})')
print(f'Difference in review count: {b_rev.sum() - c_rev.sum()}')

# Identify queries reviewed in baseline but NOT reviewed in champion
promoted = df[b_rev & (~c_rev)]
print(f'\nQueries reviewed by baseline but NOT by champion (promoted): {len(promoted)}')

# Also check queries reviewed in champion but NOT baseline
demoted = df[(~b_rev) & c_rev]
print(f'Queries NOT reviewed by baseline but reviewed by champion (demoted): {len(demoted)}')

print('\n--- Detailed inspection of the 9 Promoted Queries ---')
for idx, row in promoted.iterrows():
    q = row['query']
    task = row['task']
    b_conf = row['cand0_baseline_word_lr_t1_conf']
    b_pred = row['cand0_baseline_word_lr_t1_pred']
    b_corr = row['cand0_baseline_word_lr_t1_correct']
    c_conf = row['cand1_char_wb_lr_t1_conf']
    c_pred = row['cand1_char_wb_lr_t1_pred']
    c_corr = row['cand1_char_wb_lr_t1_correct']
    print(f'[{row["val_idx"]}] True Task: {task}')
    print(f'    Query: "{q}"')
    print(f'    Baseline: Pred={b_pred}, Conf={b_conf:.4f}, Correct={b_corr}')
    print(f'    Champion: Pred={c_pred}, Conf={c_conf:.4f}, Correct={c_corr}')

champ_errors = df[~df['cand1_char_wb_lr_t1_correct']]
print(f'\nTotal Champion Errors: {len(champ_errors)} / {len(df)}')
print('--- Champion Errors Detail ---')
for idx, row in champ_errors.iterrows():
    q = row['query']
    task = row['task']
    c_conf = row['cand1_char_wb_lr_t1_conf']
    c_pred = row['cand1_char_wb_lr_t1_pred']
    print(f'[{row["val_idx"]}] True: {task} | Pred: {c_pred} | Conf: {c_conf:.4f} | Conf < 0.65? {c_conf < tau} | Query: "{q}"')

high_conf_errors = df[(~df['cand1_char_wb_lr_t1_correct']) & (df['cand1_char_wb_lr_t1_conf'] >= tau)]
print(f'\nChampion Errors with Conf >= 0.65: {len(high_conf_errors)}')
