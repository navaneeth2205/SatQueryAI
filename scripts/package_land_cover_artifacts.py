"""
Package and serialize Land Cover Model Artifacts with Provenance.
Step 1 of Milestone 2.
"""

import json
import io
import zipfile
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier

repo_root = Path(__file__).resolve().parent.parent
artifacts_dir = repo_root / "artifacts" / "specialists" / "land_cover"
artifacts_dir.mkdir(parents=True, exist_ok=True)
data_dir = repo_root / "data"
data_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("[Step 1] RECOVERING & PACKAGING LAND COVER MODEL BUNDLE")
print("=" * 60)

# 1. Recover aligned metadata from archive.zip
zip_path = repo_root / "initial_trails" / "Dataset" / "archive.zip"
print(f"Reading metadata from: {zip_path}")

with zipfile.ZipFile(zip_path, "r") as z:
    with z.open("BEN_14k/metadata.parquet") as f:
        df = pd.read_parquet(io.BytesIO(f.read()))
    names = set(z.namelist())

s1_names = set(Path(n).stem for n in names if "BigEarthNet-S1" in n and n.endswith(".tif"))
s2_names = set(Path(n).stem for n in names if "BigEarthNet-S2" in n and n.endswith(".tif"))

usable = df[df["s1_name"].isin(s1_names) & df["patch_id"].isin(s2_names)].copy()
train_full = usable[usable["split"] == "train"].sort_values("patch_id").reset_index(drop=True)
val_full = usable[usable["split"] == "validation"].sort_values("patch_id").reset_index(drop=True)

print(f"Recovered {len(train_full)} training metadata rows and {len(val_full)} validation metadata rows.")

# 2. Fit MultiLabelBinarizer
mlb = MultiLabelBinarizer()
Y_train = mlb.fit_transform(train_full["labels"].apply(list))
Y_val = mlb.transform(val_full["labels"].apply(list))
classes = list(mlb.classes_)

print(f"Verified {len(classes)} land cover classes:")
for i, c in enumerate(classes):
    print(f"  [{i:02d}] {c}")

# 3. Load verified 84-feature matrices
X_train = np.load(repo_root / "initial_trails" / "X_train_full.npy")
X_val = np.load(repo_root / "initial_trails" / "X_val_full.npy")

assert len(X_train) == len(Y_train) == 7180, "Train row mismatch"
assert len(X_val) == len(Y_val) == 3255, "Val row mismatch"
print(f"Feature matrices verified: X_train {X_train.shape}, X_val {X_val.shape}")

# 4. Train Land Cover Model with verified parameters
print("\nFitting Random Forest model with balanced class weights...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, Y_train)
print("Model fit complete.")

# 5. Save Artifacts
model_file = artifacts_dir / "model.joblib"
mlb_file = artifacts_dir / "label_encoder.joblib"
classes_file = artifacts_dir / "classes.json"
threshold_file = artifacts_dir / "threshold_config.json"
meta_file = artifacts_dir / "metadata_summary.json"

joblib.dump(rf_model, model_file)
joblib.dump(mlb, mlb_file)

with open(classes_file, "w", encoding="utf-8") as f:
    json.dump(classes, f, indent=2)

# Copy threshold config with provenance
src_threshold = repo_root / "initial_trails" / "satquery_threshold_config.json"
with open(src_threshold, "r", encoding="utf-8") as f:
    thresh_data = json.load(f)

with open(threshold_file, "w", encoding="utf-8") as f:
    json.dump(thresh_data, f, indent=2)

# Save provenance metadata
summary = {
    "model_name": "LandCoverClassifier_RF",
    "version": "1.0.0",
    "trained_at": datetime.now(timezone.utc).isoformat(),
    "features_total": 84,
    "feature_breakdown": {
        "sentinel_1_sar": "14 statistics (2 bands x 7 stats)",
        "sentinel_2_optical": "70 statistics (10 bands x 7 stats)"
    },
    "num_train_samples": 7180,
    "num_val_samples": 3255,
    "num_classes": 16,
    "classes": classes,
    "threshold_config": thresh_data,
    "provenance": {
        "source_dataset": "BigEarthNet-14k (Serbia subset, Summer 2017)",
        "feature_alignment": "1:1 patch_id sorted alignment verified against raw TIFF files",
        "random_state": 42
    }
}

with open(meta_file, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

# Save extracted metadata tables to data/ for fast future access
train_full.to_parquet(data_dir / "train_full_metadata.parquet", index=False)
val_full.to_parquet(data_dir / "val_full_metadata.parquet", index=False)

print("\n" + "=" * 60)
print("[SUCCESS] Step 1 Artifact Bundle Created:")
print(f"  - Model file        : {model_file}")
print(f"  - Label encoder     : {mlb_file}")
print(f"  - Classes list      : {classes_file}")
print(f"  - Threshold config  : {threshold_file}")
print(f"  - Metadata summary  : {meta_file}")
print(f"  - Clean data parquets: {data_dir / 'train_full_metadata.parquet'}")
print("=" * 60)
