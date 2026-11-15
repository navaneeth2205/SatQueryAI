"""
Locked Validation Set Evaluation of Phase 1 Router Candidates.

Trained on all 109 training queries (data/starter_queries.csv, split == 'train').
Evaluated on all 56 held-out validation queries (split == 'val').

Reports both Tier 1 (Raw Statistical ML Pipeline) and Tier 2 (Full System with Keyword Guard).
"""

import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    log_loss
)
from satquery.router import _contains_task_keywords
from scripts.benchmark_phase1_candidates import compute_multiclass_brier, compute_ece_10, apply_linguistic_keyword_guard

df = pd.read_csv(repo_root / "data" / "starter_queries.csv")
train_df = df[df["split"] == "train"].reset_index(drop=True)
val_df = df[df["split"] == "val"].reset_index(drop=True)

X_train = np.array(train_df["query"].tolist(), dtype=object)
y_train = np.array(train_df["task"].tolist(), dtype=str)
X_val = np.array(val_df["query"].tolist(), dtype=object)
y_val = np.array(val_df["task"].tolist(), dtype=str)

classes = np.array(sorted(train_df["task"].unique()))
target_tasks = list(classes)

# Define candidates
candidate_factories = {
    "cand0_baseline_word_lr": lambda: Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", strip_accents="unicode", lowercase=True)),
        ("clf", CalibratedClassifierCV(estimator=LogisticRegression(C=1.0, class_weight="balanced", random_state=42, max_iter=1000), method="sigmoid", cv=3))
    ]),
    "cand1_char_wb_lr": lambda: Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, strip_accents="unicode", lowercase=True)),
        ("clf", CalibratedClassifierCV(estimator=LogisticRegression(C=1.0, class_weight="balanced", random_state=42, max_iter=1000), method="sigmoid", cv=3))
    ]),
    "cand3_word_char_svm": lambda: Pipeline([
        ("feats", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", strip_accents="unicode", lowercase=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, strip_accents="unicode", lowercase=True))
        ])),
        ("clf", CalibratedClassifierCV(estimator=LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=2000), method="sigmoid", cv=3))
    ]),
    "cand5_dense_ngram_tuned_lr": lambda: Pipeline([
        ("feats", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, stop_words="english", strip_accents="unicode", min_df=1, lowercase=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, strip_accents="unicode", lowercase=True))
        ])),
        ("clf", CalibratedClassifierCV(estimator=LogisticRegression(C=2.0, class_weight="balanced", random_state=42, max_iter=1500), method="sigmoid", cv=3))
    ])
}

val_results = {}
val_predictions_master = val_df[["query", "task"]].copy()
val_predictions_master["val_idx"] = val_df.index

print("=" * 80)
print(f"EVALUATING PHASE 1 CANDIDATES ON LOCKED VALIDATION SET (N={len(val_df)})")
print("=" * 80)

for cand_name, factory in candidate_factories.items():
    pipe = factory()
    pipe.fit(X_train, y_train)
    
    probs = pipe.predict_proba(X_val)
    preds_t1 = classes[np.argmax(probs, axis=1)]
    confs_t1 = np.max(probs, axis=1)
    
    # Tier 1 Metrics
    acc_t1 = accuracy_score(y_val, preds_t1)
    prec_t1, rec_t1, f1_t1, _ = precision_recall_fscore_support(y_val, preds_t1, average="macro", zero_division=0)
    brier_t1 = compute_multiclass_brier(y_val, probs, classes)
    nll_t1 = log_loss(y_val, probs, labels=classes)
    ece_t1, ece_bins_t1 = compute_ece_10(y_val, probs, classes)
    review_t1 = float(np.mean(confs_t1 < 0.65))
    cm_t1 = confusion_matrix(y_val, preds_t1, labels=target_tasks).tolist()
    rep_t1 = classification_report(y_val, preds_t1, labels=target_tasks, output_dict=True, zero_division=0)

    # Tier 2 (with frozen keyword guard)
    preds_t2 = []
    confs_t2 = []
    guards_t2 = []
    for q_text, p_t1, c_t1, p_vec in zip(X_val, preds_t1, confs_t1, probs):
        p_dict = {cls: float(p) for cls, p in zip(classes, p_vec)}
        p_t2, c_t2, g_flag = apply_linguistic_keyword_guard(q_text, p_t1, float(c_t1), p_dict, classes, threshold=0.65)
        preds_t2.append(p_t2)
        confs_t2.append(c_t2)
        guards_t2.append(g_flag)
        
    preds_t2 = np.array(preds_t2)
    confs_t2 = np.array(confs_t2)
    
    acc_t2 = accuracy_score(y_val, preds_t2)
    prec_t2, rec_t2, f1_t2, _ = precision_recall_fscore_support(y_val, preds_t2, average="macro", zero_division=0)
    review_t2 = float(np.mean(confs_t2 < 0.65))
    cm_t2 = confusion_matrix(y_val, preds_t2, labels=target_tasks).tolist()
    rep_t2 = classification_report(y_val, preds_t2, labels=target_tasks, output_dict=True, zero_division=0)

    val_results[cand_name] = {
        "tier1_statistical": {
            "accuracy": float(acc_t1),
            "macro_f1": float(f1_t1),
            "macro_precision": float(prec_t1),
            "macro_recall": float(rec_t1),
            "brier": float(brier_t1),
            "nll": float(nll_t1),
            "ece_10": float(ece_t1),
            "review_rate": float(review_t1),
            "mean_confidence": float(np.mean(confs_t1)),
            "confusion_matrix": cm_t1,
            "classification_report": rep_t1
        },
        "tier2_full_system": {
            "accuracy": float(acc_t2),
            "macro_f1": float(f1_t2),
            "macro_precision": float(prec_t2),
            "macro_recall": float(rec_t2),
            "review_rate": float(review_t2),
            "guard_triggers": int(np.sum(guards_t2)),
            "confusion_matrix": cm_t2,
            "classification_report": rep_t2
        }
    }

    val_predictions_master[f"{cand_name}_t1_pred"] = preds_t1
    val_predictions_master[f"{cand_name}_t1_conf"] = np.round(confs_t1, 4)
    val_predictions_master[f"{cand_name}_t1_correct"] = (preds_t1 == y_val)
    val_predictions_master[f"{cand_name}_t2_pred"] = preds_t2
    val_predictions_master[f"{cand_name}_t2_correct"] = (preds_t2 == y_val)

    print(f"\n[{cand_name}]")
    print(f"  Tier 1 (Raw Statistical): Macro F1: {f1_t1*100:.2f}% | Accuracy: {acc_t1*100:.2f}% ({sum(preds_t1 == y_val)}/56) | Brier: {brier_t1:.4f} | Review: {review_t1*100:.1f}%")
    print(f"  Tier 2 (With Guard):     Macro F1: {f1_t2*100:.2f}% | Accuracy: {acc_t2*100:.2f}% ({sum(preds_t2 == y_val)}/56) | Triggers: {np.sum(guards_t2)}")

# Save validation results
out_dir = repo_root / "artifacts" / "phase1_router"
val_res_path = out_dir / "phase1_locked_validation_results.json"
with open(val_res_path, "w", encoding="utf-8") as f:
    json.dump(val_results, f, indent=2)

val_pred_path = out_dir / "phase1_locked_validation_predictions.csv"
val_predictions_master.to_csv(val_pred_path, index=False)
val_predictions_master.to_csv(r"C:\Users\B NAVANEETH\.gemini\antigravity\brain\89b5d681-0c9a-403e-9edc-f4bd3eee0aba\scratch\phase1_locked_validation_predictions.csv", index=False)

print("\n" + "=" * 80)
print(f"Locked Validation Evaluation Complete! Saved to:\n  - {val_res_path}\n  - {val_pred_path}")
print("=" * 80)
