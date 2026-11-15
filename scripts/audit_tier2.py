import pandas as pd

df = pd.read_csv(r'c:\Users\B NAVANEETH\SatQueryAI\artifacts\phase1_router\phase1_locked_validation_predictions.csv')

print('=== Tier 2 Errors for cand1_char_wb_lr ===')
t2_errors = df[~df['cand1_char_wb_lr_t2_correct']]
print(f'Total Tier 2 Errors: {len(t2_errors)} / {len(df)}')
for idx, row in t2_errors.iterrows():
    print(f'[{row["val_idx"]}] True: {row["task"]} | T2 Pred: {row["cand1_char_wb_lr_t2_pred"]} | T1 Conf: {row["cand1_char_wb_lr_t1_conf"]:.4f} | Query: "{row["query"]}"')

print('\nCheck Query [3] specifically across all columns:')
row3 = df.loc[3]
for col in df.columns:
    print(f'  {col}: {row3[col]}')
