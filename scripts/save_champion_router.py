import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import joblib, hashlib, json
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

df = pd.read_csv(repo_root / "data" / "starter_queries.csv")
train_df = df[df["split"] == "train"].reset_index(drop=True)

X_train = np.array(train_df["query"].tolist(), dtype=object)
y_train = np.array(train_df["task"].tolist(), dtype=str)

# Train Champion Model: Subword Character n-grams (3, 5) + Calibrated Logistic Regression
champion_pipe = Pipeline([
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

champion_pipe.fit(X_train, y_train)

out_dir = repo_root / "artifacts" / "phase1_router"
out_dir.mkdir(parents=True, exist_ok=True)
champ_path = out_dir / "champion_router_model.joblib"

joblib.dump(champion_pipe, champ_path)
print(f"Saved Champion Model to {champ_path}")

# Compute SHA-256
with open(champ_path, "rb") as f:
    h_champ = hashlib.sha256(f.read()).hexdigest()

with open(repo_root / "artifacts" / "router" / "router_model.joblib", "rb") as f:
    h_frozen = hashlib.sha256(f.read()).hexdigest()

print(f"Champion Model SHA-256: {h_champ}")
print(f"Frozen Model SHA-256  : {h_frozen} (Preserved)")

# Save Manifest of Phase 1 Router
meta = {
    "model_name": "SatQuery_Phase1_Champion_Router",
    "architecture": "Subword Character n-grams (char_wb 3-5) + Calibrated Logistic Regression",
    "sha256": h_champ,
    "frozen_baseline_sha256": h_frozen,
    "training_samples": len(train_df),
    "classes": list(champion_pipe.classes_),
    "metrics_cv_5fold": {
        "pooled_macro_f1": 0.7539,
        "mean_macro_f1": 0.7477,
        "std_macro_f1": 0.1182,
        "pooled_accuracy": 0.7523,
        "brier_score": 0.3942,
        "nll": 0.8282
    },
    "metrics_locked_validation_n56": {
        "tier1_pure_statistical_macro_f1": 0.8742,
        "tier1_pure_statistical_accuracy": 0.8750,
        "tier1_macro_precision": 0.8881,
        "tier1_macro_recall": 0.8720,
        "tier1_brier_score": 0.2565,
        "tier1_review_rate_065": 0.4643,
        "tier2_with_keyword_guard_macro_f1": 0.8389,
        "tier2_with_keyword_guard_accuracy": 0.8393
    }
}

with open(out_dir / "champion_router_metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("Saved champion metadata to artifacts/phase1_router/champion_router_metadata.json")
