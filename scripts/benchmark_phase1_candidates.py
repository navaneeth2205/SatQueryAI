"""
Phase 1 Candidate Router Benchmark Script for SatQuery AI.

Evaluates candidate ML architectures using the exact 5-fold stratified cross-validation
manifest defined in configs/phase1_cv_split_manifest.json.

Strict Safeguards:
- Preprocessing and vectorizers fit STRICTLY inside each training fold.
- Probability calibration fit STRICTLY on training fold data (nested 3-fold CV).
- Evaluates both Tier 1 (Pure Statistical ML) and Tier 2 (Full System with Keyword Guard).
- Exports fold-level metrics, summary mean/std, pooled out-of-fold predictions, and calibration curves.
"""

import sys
from pathlib import Path

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    log_loss,
    brier_score_loss
)
from satquery.router import _contains_task_keywords


def compute_multiclass_brier(y_true, y_prob, classes):
    """Compute multi-class Brier score: mean squared difference from one-hot indicator."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n_samples, n_classes = y_prob.shape
    one_hot = np.zeros((n_samples, n_classes))
    for i, label in enumerate(y_true):
        c_idx = np.where(classes == label)[0][0]
        one_hot[i, c_idx] = 1.0
    return float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1)))


def compute_ece_10(y_true, y_prob, classes, n_bins=10):
    """Compute 10-bin Expected Calibration Error on predictions."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    confidences = np.max(y_prob, axis=1)
    predictions = classes[np.argmax(y_prob, axis=1)]
    accuracies = (predictions == y_true).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_details = []
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = float(np.mean(in_bin))
        if prop_in_bin > 0:
            accuracy_in_bin = float(np.mean(accuracies[in_bin]))
            avg_confidence_in_bin = float(np.mean(confidences[in_bin]))
            abs_diff = abs(accuracy_in_bin - avg_confidence_in_bin)
            ece += abs_diff * prop_in_bin
            bin_details.append({
                "bin": f"({bin_boundaries[i]:.2f}, {bin_boundaries[i+1]:.2f}]",
                "count": int(np.sum(in_bin)),
                "acc": round(accuracy_in_bin, 4),
                "conf": round(avg_confidence_in_bin, 4),
                "gap": round(abs_diff, 4)
            })
        else:
            bin_details.append({
                "bin": f"({bin_boundaries[i]:.2f}, {bin_boundaries[i+1]:.2f}]",
                "count": 0,
                "acc": 0.0,
                "conf": 0.0,
                "gap": 0.0
            })
            
    return float(ece), bin_details


def apply_linguistic_keyword_guard(query, text_task, text_conf, prob_dict, classes, threshold=0.65):
    """Apply the frozen linguistic keyword guard rule (Tier 2 dispatch)."""
    top_has_kw = _contains_task_keywords(text_task, query)
    if not top_has_kw and text_conf < threshold:
        for cand_task in classes:
            if cand_task != text_task and _contains_task_keywords(str(cand_task), query):
                return str(cand_task), float(prob_dict.get(cand_task, text_conf)), True
    return text_task, text_conf, False


def build_candidate_pipelines():
    """Return dictionary of candidate model builders."""
    candidates = {}

    # Candidate 0: Frozen Baseline ML Pipeline
    candidates["cand0_baseline_word_lr"] = lambda: Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english",
            strip_accents="unicode",
            lowercase=True
        )),
        ("clf", CalibratedClassifierCV(
            estimator=LogisticRegression(C=1.0, class_weight="balanced", random_state=42, max_iter=1000),
            method="sigmoid",
            cv=3
        ))
    ])

    # Candidate 1: Subword Character n-grams (3, 5) with word boundaries
    candidates["cand1_char_wb_lr"] = lambda: Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True
        )),
        ("clf", CalibratedClassifierCV(
            estimator=LogisticRegression(C=1.0, class_weight="balanced", random_state=42, max_iter=1000),
            method="sigmoid",
            cv=3
        ))
    ])

    # Candidate 2: Word (1, 2) + Char_WB (3, 5) FeatureUnion + Logistic Regression
    candidates["cand2_word_char_union_lr"] = lambda: Pipeline([
        ("feats", FeatureUnion([
            ("word", TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                stop_words="english",
                strip_accents="unicode",
                lowercase=True
            )),
            ("char", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                sublinear_tf=True,
                strip_accents="unicode",
                lowercase=True
            ))
        ])),
        ("clf", CalibratedClassifierCV(
            estimator=LogisticRegression(C=1.0, class_weight="balanced", random_state=42, max_iter=1000),
            method="sigmoid",
            cv=3
        ))
    ])

    # Candidate 3: Word + Char FeatureUnion + Calibrated Linear SVM (LinearSVC)
    candidates["cand3_word_char_svm"] = lambda: Pipeline([
        ("feats", FeatureUnion([
            ("word", TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                stop_words="english",
                strip_accents="unicode",
                lowercase=True
            )),
            ("char", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                sublinear_tf=True,
                strip_accents="unicode",
                lowercase=True
            ))
        ])),
        ("clf", CalibratedClassifierCV(
            estimator=LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=2000),
            method="sigmoid",
            cv=3
        ))
    ])

    # Candidate 4: Word + Char FeatureUnion + Calibrated Ridge Classifier
    candidates["cand4_word_char_ridge"] = lambda: Pipeline([
        ("feats", FeatureUnion([
            ("word", TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                stop_words="english",
                strip_accents="unicode",
                lowercase=True
            )),
            ("char", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                sublinear_tf=True,
                strip_accents="unicode",
                lowercase=True
            ))
        ])),
        ("clf", CalibratedClassifierCV(
            estimator=RidgeClassifier(alpha=1.0, class_weight="balanced", random_state=42),
            method="sigmoid",
            cv=3
        ))
    ])

    # Candidate 5: Word (1, 3) + Char (2, 5) FeatureUnion + Tuned Logistic Regression (C=2.0)
    candidates["cand5_dense_ngram_tuned_lr"] = lambda: Pipeline([
        ("feats", FeatureUnion([
            ("word", TfidfVectorizer(
                ngram_range=(1, 3),
                sublinear_tf=True,
                stop_words="english",
                strip_accents="unicode",
                min_df=1,
                lowercase=True
            )),
            ("char", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 5),
                sublinear_tf=True,
                strip_accents="unicode",
                lowercase=True
            ))
        ])),
        ("clf", CalibratedClassifierCV(
            estimator=LogisticRegression(C=2.0, class_weight="balanced", random_state=42, max_iter=1500),
            method="sigmoid",
            cv=3
        ))
    ])

    return candidates


def run_phase1_benchmark():
    data_path = repo_root / "data" / "starter_queries.csv"
    manifest_path = repo_root / "configs" / "phase1_cv_split_manifest.json"
    
    df = pd.read_csv(data_path)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        split_manifest = json.load(f)

    candidates = build_candidate_pipelines()
    
    out_dir = repo_root / "artifacts" / "phase1_router"
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark_results = {}
    oof_predictions_master = train_df[["query", "task"]].copy()
    oof_predictions_master["query_idx"] = train_df.index

    classes = np.array(sorted(train_df["task"].unique()))
    target_tasks = list(classes)

    print("=" * 80)
    print("SATQUERY AI — PHASE 1 INTENT ROUTER 5-FOLD CROSS-VALIDATION BENCHMARK")
    print(f"Training Samples: {len(train_df)} across {len(classes)} classes")
    print(f"Candidates to Evaluate: {len(candidates)}")
    print("=" * 80)

    for cand_name, pipe_factory in candidates.items():
        print(f"\n>>> Running Evaluation for Candidate: {cand_name} ...")
        
        fold_metrics_tier1 = []
        fold_metrics_tier2 = []
        
        oof_preds_tier1 = [None] * len(train_df)
        oof_probs_tier1 = [None] * len(train_df)
        oof_confs_tier1 = [None] * len(train_df)
        
        oof_preds_tier2 = [None] * len(train_df)
        oof_confs_tier2 = [None] * len(train_df)
        oof_guard_flags = [False] * len(train_df)

        for fold_num in range(1, 6):
            fold_key = f"fold_{fold_num}"
            train_idx = split_manifest["folds"][fold_key]["train_indices"]
            val_idx = split_manifest["folds"][fold_key]["val_indices"]
            
            X_train = np.array(train_df.iloc[train_idx]["query"].tolist(), dtype=object)
            y_train = np.array(train_df.iloc[train_idx]["task"].tolist(), dtype=str)
            X_val = np.array(train_df.iloc[val_idx]["query"].tolist(), dtype=object)
            y_val = np.array(train_df.iloc[val_idx]["task"].tolist(), dtype=str)
            
            # Construct and fit fresh candidate pipeline strictly inside this fold
            pipe = pipe_factory()
            pipe.fit(X_train, y_train)
            
            # Tier 1: Raw statistical pipeline inference
            probs_val = pipe.predict_proba(X_val)
            preds_val_t1 = classes[np.argmax(probs_val, axis=1)]
            confs_val_t1 = np.max(probs_val, axis=1)
            
            # Tier 2: Linguistic keyword guard applied to sub-threshold predictions
            preds_val_t2 = []
            confs_val_t2 = []
            guards_val = []
            
            for q_text, p_t1, c_t1, prob_vec in zip(X_val, preds_val_t1, confs_val_t1, probs_val):
                p_dict = {cls: float(p) for cls, p in zip(classes, prob_vec)}
                p_t2, c_t2, g_flag = apply_linguistic_keyword_guard(
                    query=q_text,
                    text_task=p_t1,
                    text_conf=float(c_t1),
                    prob_dict=p_dict,
                    classes=classes,
                    threshold=0.65
                )
                preds_val_t2.append(p_t2)
                confs_val_t2.append(c_t2)
                guards_val.append(g_flag)

            preds_val_t2 = np.array(preds_val_t2)
            confs_val_t2 = np.array(confs_val_t2)

            # Record out-of-fold predictions
            for idx_in_fold, global_idx in enumerate(val_idx):
                oof_preds_tier1[global_idx] = preds_val_t1[idx_in_fold]
                oof_probs_tier1[global_idx] = probs_val[idx_in_fold]
                oof_confs_tier1[global_idx] = confs_val_t1[idx_in_fold]
                
                oof_preds_tier2[global_idx] = preds_val_t2[idx_in_fold]
                oof_confs_tier2[global_idx] = confs_val_t2[idx_in_fold]
                oof_guard_flags[global_idx] = guards_val[idx_in_fold]

            # Fold-level metrics for Tier 1
            acc_t1 = accuracy_score(y_val, preds_val_t1)
            prec_t1, rec_t1, f1_t1, _ = precision_recall_fscore_support(y_val, preds_val_t1, average="macro", zero_division=0)
            nll_t1 = log_loss(y_val, probs_val, labels=classes)
            brier_t1 = compute_multiclass_brier(y_val, probs_val, classes)
            review_rate_t1 = float(np.mean(confs_val_t1 < 0.65))

            fold_metrics_tier1.append({
                "fold": fold_num,
                "accuracy": float(acc_t1),
                "macro_f1": float(f1_t1),
                "macro_precision": float(prec_t1),
                "macro_recall": float(rec_t1),
                "nll": float(nll_t1),
                "brier": float(brier_t1),
                "review_rate": float(review_rate_t1),
                "mean_confidence": float(np.mean(confs_val_t1))
            })

            # Fold-level metrics for Tier 2
            acc_t2 = accuracy_score(y_val, preds_val_t2)
            prec_t2, rec_t2, f1_t2, _ = precision_recall_fscore_support(y_val, preds_val_t2, average="macro", zero_division=0)
            review_rate_t2 = float(np.mean(confs_val_t2 < 0.65))

            fold_metrics_tier2.append({
                "fold": fold_num,
                "accuracy": float(acc_t2),
                "macro_f1": float(f1_t2),
                "macro_precision": float(prec_t2),
                "macro_recall": float(rec_t2),
                "review_rate": float(review_rate_t2),
                "mean_confidence": float(np.mean(confs_val_t2))
            })

        # Aggregated Out-Of-Fold Analysis
        oof_preds_t1_arr = np.array(oof_preds_tier1)
        oof_probs_t1_arr = np.array(oof_probs_tier1)
        oof_confs_t1_arr = np.array(oof_confs_tier1)
        y_true_all = train_df["task"].values

        oof_preds_t2_arr = np.array(oof_preds_tier2)
        oof_confs_t2_arr = np.array(oof_confs_tier2)

        # Pooled Tier 1 metrics
        pooled_acc_t1 = accuracy_score(y_true_all, oof_preds_t1_arr)
        pooled_prec_t1, pooled_rec_t1, pooled_f1_t1, _ = precision_recall_fscore_support(y_true_all, oof_preds_t1_arr, average="macro", zero_division=0)
        pooled_nll_t1 = log_loss(y_true_all, oof_probs_t1_arr, labels=classes)
        pooled_brier_t1 = compute_multiclass_brier(y_true_all, oof_probs_t1_arr, classes)
        pooled_ece_t1, bin_details_t1 = compute_ece_10(y_true_all, oof_probs_t1_arr, classes)
        pooled_review_t1 = float(np.mean(oof_confs_t1_arr < 0.65))

        # Pooled Tier 2 metrics
        pooled_acc_t2 = accuracy_score(y_true_all, oof_preds_t2_arr)
        pooled_prec_t2, pooled_rec_t2, pooled_f1_t2, _ = precision_recall_fscore_support(y_true_all, oof_preds_t2_arr, average="macro", zero_division=0)
        pooled_review_t2 = float(np.mean(oof_confs_t2_arr < 0.65))

        # Per-class Tier 1 metrics
        per_class_report_t1 = classification_report(y_true_all, oof_preds_t1_arr, labels=target_tasks, output_dict=True, zero_division=0)
        cm_t1 = confusion_matrix(y_true_all, oof_preds_t1_arr, labels=target_tasks).tolist()

        # Fold mean & std
        df_f1 = pd.DataFrame(fold_metrics_tier1)
        df_f2 = pd.DataFrame(fold_metrics_tier2)

        benchmark_results[cand_name] = {
            "tier1_statistical": {
                "per_fold": fold_metrics_tier1,
                "mean_macro_f1": float(df_f1["macro_f1"].mean()),
                "std_macro_f1": float(df_f1["macro_f1"].std()),
                "mean_accuracy": float(df_f1["accuracy"].mean()),
                "std_accuracy": float(df_f1["accuracy"].std()),
                "mean_precision": float(df_f1["macro_precision"].mean()),
                "std_precision": float(df_f1["macro_precision"].std()),
                "mean_recall": float(df_f1["macro_recall"].mean()),
                "std_recall": float(df_f1["macro_recall"].std()),
                "mean_brier": float(df_f1["brier"].mean()),
                "std_brier": float(df_f1["brier"].std()),
                "mean_nll": float(df_f1["nll"].mean()),
                "std_nll": float(df_f1["nll"].std()),
                "mean_review_rate": float(df_f1["review_rate"].mean()),
                "std_review_rate": float(df_f1["review_rate"].std()),
                "pooled_oof": {
                    "accuracy": float(pooled_acc_t1),
                    "macro_f1": float(pooled_f1_t1),
                    "macro_precision": float(pooled_prec_t1),
                    "macro_recall": float(pooled_rec_t1),
                    "nll": float(pooled_nll_t1),
                    "brier": float(pooled_brier_t1),
                    "ece_10": float(pooled_ece_t1),
                    "ece_bins": bin_details_t1,
                    "review_rate": float(pooled_review_t1),
                    "mean_confidence": float(np.mean(oof_confs_t1_arr)),
                    "confusion_matrix": cm_t1,
                    "per_class_report": per_class_report_t1
                }
            },
            "tier2_full_system": {
                "per_fold": fold_metrics_tier2,
                "mean_macro_f1": float(df_f2["macro_f1"].mean()),
                "std_macro_f1": float(df_f2["macro_f1"].std()),
                "mean_accuracy": float(df_f2["accuracy"].mean()),
                "std_accuracy": float(df_f2["accuracy"].std()),
                "pooled_oof": {
                    "accuracy": float(pooled_acc_t2),
                    "macro_f1": float(pooled_f1_t2),
                    "macro_precision": float(pooled_prec_t2),
                    "macro_recall": float(pooled_rec_t2),
                    "review_rate": float(pooled_review_t2),
                    "guard_trigger_count": int(np.sum(oof_guard_flags))
                }
            }
        }

        # Add to master OOF DataFrame
        oof_predictions_master[f"{cand_name}_t1_pred"] = oof_preds_t1_arr
        oof_predictions_master[f"{cand_name}_t1_conf"] = np.round(oof_confs_t1_arr, 4)
        oof_predictions_master[f"{cand_name}_t1_correct"] = (oof_preds_t1_arr == y_true_all)
        oof_predictions_master[f"{cand_name}_t2_pred"] = oof_preds_t2_arr
        oof_predictions_master[f"{cand_name}_t2_correct"] = (oof_preds_t2_arr == y_true_all)

        print(f"  Tier 1 Mean Macro F1: {df_f1['macro_f1'].mean()*100:.2f}% ± {df_f1['macro_f1'].std()*100:.2f}% | Pooled Macro F1: {pooled_f1_t1*100:.2f}%")
        print(f"  Tier 1 Mean Accuracy: {df_f1['accuracy'].mean()*100:.2f}% ± {df_f1['accuracy'].std()*100:.2f}% | Pooled Accuracy: {pooled_acc_t1*100:.2f}%")
        print(f"  Tier 1 Pooled Brier : {pooled_brier_t1:.4f} | NLL: {pooled_nll_t1:.4f} | ECE: {pooled_ece_t1:.4f} | Review Rate: {pooled_review_t1*100:.1f}%")
        print(f"  Tier 2 Pooled Macro F1 (with guard): {pooled_f1_t2*100:.2f}% | Acc: {pooled_acc_t2*100:.2f}% | Guard Triggers: {np.sum(oof_guard_flags)}")

    # Save benchmark results
    results_path = out_dir / "phase1_candidate_cv_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)

    # Save OOF predictions
    oof_path = out_dir / "phase1_pooled_oof_predictions.csv"
    oof_predictions_master.to_csv(oof_path, index=False)
    oof_predictions_master.to_csv(r"C:\Users\B NAVANEETH\.gemini\antigravity\brain\89b5d681-0c9a-403e-9edc-f4bd3eee0aba\scratch\phase1_pooled_oof_predictions.csv", index=False)

    print("\n" + "=" * 80)
    print(f"Phase 1 CV Benchmark Complete! Results saved to:\n  - {results_path}\n  - {oof_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_phase1_benchmark()
