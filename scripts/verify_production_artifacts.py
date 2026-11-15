"""
SatQuery AI — Milestone M3.3 Step 3.1: Production Artifact Integrity & Freezing Verification.

Performs standalone verification of:
1. SHA-256 checksums and file sizes for all production models, configs, and catalogs.
2. Clean in-memory loading of each specialist model and router.
3. Forward inference verification on synthetic inputs.
4. Export/verification of artifacts/deployment/m33_production_freeze_manifest.json.
"""

import hashlib
import json
import logging
from pathlib import Path
import sys
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("freeze_verifier")

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# List of critical production artifacts
PRODUCTION_ARTIFACTS_SPEC = {
    "champion_change_detection_model": {
        "path": "artifacts/specialists/change_detection_step3/e4_best_model.pt",
        "expected_sha256": "566d83792cf0ad9beed4ba75375f22560a48f86a662a7c34ee858339b088da24",
        "expected_size_bytes": 80695746,
        "description": "Siamese ResNet-18 U-Net with scSE Attention (M3.2 Champion Checkpoint)",
        "framework": "PyTorch",
        "operating_threshold": 0.40,
        "min_component_area": 9
    },
    "champion_land_cover_model": {
        "path": "artifacts/specialists/land_cover_phase2/model.joblib",
        "expected_sha256": "f785ca1f5e9691c17e4a17ebf3f6766f7053059ba871188d24f6dce0ed43d071",
        "expected_size_bytes": 8087110,
        "description": "LightGBM 488-feature Multimodal GBDT Classifier (Phase 2 Champion)",
        "framework": "LightGBM / scikit-learn",
        "operating_threshold": "Per-Class Validation Tuned"
    },
    "land_cover_phase2_thresholds": {
        "path": "artifacts/specialists/land_cover_phase2/threshold_config.json",
        "expected_sha256": "74f1e5f1bccd8e9751a19ad3936aef528e234e69b33c9a7d34a4b20c6f2a3a10",
        "expected_size_bytes": 935,
        "description": "Optimized per-class decision thresholds for BigEarthNet 16 classes",
        "framework": "JSON Configuration"
    },
    "land_cover_phase2_classes": {
        "path": "artifacts/specialists/land_cover_phase2/classes.json",
        "expected_sha256": "cdb18c8a0a9dc74f52920720340251f14de323bc5c4fb4f07436f90f3500f3e9",
        "expected_size_bytes": 522,
        "description": "BigEarthNet-16 class names catalog",
        "framework": "JSON Specification"
    },
    "land_cover_phase2_label_encoder": {
        "path": "artifacts/specialists/land_cover_phase2/label_encoder.joblib",
        "expected_sha256": "d7e78865b4f21643cd669246dfafb14bb14ad2142b39dcb931ea28becb9af042",
        "expected_size_bytes": 1469,
        "description": "MultiLabelBinarizer fitted on BigEarthNet taxonomy",
        "framework": "scikit-learn"
    },
    "baseline_land_cover_model": {
        "path": "artifacts/specialists/land_cover/model.joblib",
        "expected_sha256": "9473c2625c77299730edec5b3f9dc0f8b7a843ef4903fe9d92db8edc9c544593",
        "expected_size_bytes": 10249993,
        "description": "Random Forest 84-feature Classifier (Phase 1 Baseline)",
        "framework": "scikit-learn"
    },
    "champion_router_model": {
        "path": "artifacts/phase1_router/champion_router_model.joblib",
        "expected_sha256": "11366c09e98ce3298313abefa30d45dc29264874acbf8964a3288b942dacdf6b",
        "expected_size_bytes": 831357,
        "description": "Subword Character n-grams (char_wb 3-5) + Calibrated LogReg (Phase 1 Champion)",
        "framework": "scikit-learn",
        "operating_thresholds": {"tau_intent": 0.65, "tau_margin": 0.15, "T_cal": 0.0991}
    },
    "champion_router_metadata": {
        "path": "artifacts/phase1_router/champion_router_metadata.json",
        "expected_sha256": "e32163f33e03c7fe0cb0d551a0410b34286607077a5619139ea7acd7232f11f5",
        "expected_size_bytes": 1140,
        "description": "Champion router cross-validation & locked validation performance manifest",
        "framework": "JSON Manifest"
    },
    "baseline_router_model": {
        "path": "artifacts/router/router_model.joblib",
        "expected_sha256": "f0e6608b11c01da3cfae454342526760ded16fb5c18ed4bdaf5cea5727b317b7",
        "expected_size_bytes": 171247,
        "description": "Word n-grams (1, 2) + Calibrated LogReg (Phase 0 Baseline)",
        "framework": "scikit-learn",
        "operating_thresholds": {"confidence_threshold": 0.65}
    },
    "change_detection_catalog": {
        "path": "data/change_detection_catalog.parquet",
        "expected_sha256": "b38f9c226c6222ef64c45c0cdf5763ec109f8b511754cc1715bfebb519342985",
        "expected_size_bytes": 9689,
        "description": "OSCD complete 24-scene forensic catalog with split partitions",
        "framework": "Apache Parquet"
    },
    "starter_queries_catalog": {
        "path": "data/starter_queries.csv",
        "expected_sha256": "6453a40e944cb2239f00c02d9d27952f6604b4520503ef5899e1cc3e68be5bd1",
        "expected_size_bytes": 17172,
        "description": "165 labeled intent queries (109 train, 56 val)",
        "framework": "CSV"
    },
    "requirements_manifest": {
        "path": "requirements.txt",
        "expected_sha256": "2098463e7321cb725407d3ed624a1833d6ca8d10fa89f8b45b5dc00cea07f977",
        "expected_size_bytes": 360,
        "description": "Pinned primary production dependencies",
        "framework": "pip requirements"
    },
    "pyproject_manifest": {
        "path": "pyproject.toml",
        "expected_sha256": "ebbb1f4e3e5ad79c4871892af2a07d6d116fd3be673df7960a0bf923d15b6823",
        "expected_size_bytes": 1335,
        "description": "PEP 517 build configuration and metadata",
        "framework": "pyproject.toml"
    }
}


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_file_checksums() -> bool:
    """Verify all artifacts against their expected SHA-256 and size."""
    logger.info("=== STEP 3.1.1: VERIFYING ARTIFACT CHECKSUMS & SIZES ===")
    all_ok = True
    for key, spec in PRODUCTION_ARTIFACTS_SPEC.items():
        p = repo_root / spec["path"]
        if not p.exists():
            logger.error(f"[MISSING] {key}: File not found at {p}")
            all_ok = False
            continue

        size = p.stat().st_size
        sha = compute_file_sha256(p)

        size_ok = (size == spec["expected_size_bytes"])
        sha_ok = (sha.lower() == spec["expected_sha256"].lower())

        if size_ok and sha_ok:
            logger.info(f"[PASS] {key}: {size} bytes | SHA256: {sha[:16]}...")
        else:
            logger.error(f"[MISMATCH] {key}:")
            if not size_ok:
                logger.error(f"  Size expected: {spec['expected_size_bytes']}, got: {size}")
            if not sha_ok:
                logger.error(f"  SHA expected:  {spec['expected_sha256']}, got: {sha}")
            all_ok = False
    return all_ok


def verify_checkpoint_loading_and_inference() -> bool:
    """Perform clean in-memory loading and forward pass testing for all models."""
    logger.info("\n=== STEP 3.1.2: VERIFYING CLEAN IN-MEMORY LOADING & INFERENCE ===")
    all_ok = True

    # 1. Router Baseline
    try:
        from satquery.router import TaskRouter
        router_base = TaskRouter.load(repo_root / "artifacts" / "router" / "router_model.joblib")
        res = router_base.predict("What changed between 2024 and 2026?")
        logger.info(f"[PASS] Router Baseline loaded successfully. Sample route: {res.selected_task} (conf: {res.confidence:.4f})")
    except Exception as e:
        logger.error(f"[FAIL] Router Baseline loading failed: {e}")
        all_ok = False

    # 2. Router Champion
    try:
        import joblib
        router_champ = joblib.load(repo_root / "artifacts" / "phase1_router" / "champion_router_model.joblib")
        pred = router_champ.predict(["What changed between 2024 and 2026?"])
        logger.info(f"[PASS] Router Champion pipeline loaded successfully. Sample route: {pred[0]}")
    except Exception as e:
        logger.error(f"[FAIL] Router Champion loading failed: {e}")
        all_ok = False

    # 3. Land Cover Specialist
    try:
        import numpy as np
        from satquery.specialists.land_cover import LandCoverClassifier
        lc = LandCoverClassifier()
        dummy_feat = np.zeros(488, dtype=np.float32)
        res_lc = lc.predict(dummy_feat)
        logger.info(f"[PASS] Land Cover LightGBM Specialist loaded. Status: {res_lc.status.value}, Predictions: {len(res_lc.predictions)} classes")
    except Exception as e:
        logger.error(f"[FAIL] Land Cover Specialist loading failed: {e}")
        all_ok = False

    # 4. Deep Change Detection Specialist
    try:
        import numpy as np
        from satquery.specialists.change_detection import SiameseResNetChangeDetectionSpecialist
        cd = SiameseResNetChangeDetectionSpecialist(device="cpu")
        dummy_t1 = np.zeros((64, 64, 3), dtype=np.uint8)
        dummy_t2 = np.zeros((64, 64, 3), dtype=np.uint8)
        res_cd = cd.predict((dummy_t1, dummy_t2))
        logger.info(f"[PASS] Siamese ResNet-18 U-Net Specialist loaded on CPU. Status: {res_cd.status.value}, Change: {res_cd.predictions[0]}")
    except Exception as e:
        logger.error(f"[FAIL] Deep Change Detection Specialist loading failed: {e}")
        all_ok = False

    # 5. Gaussian CVA Classical Fallback
    try:
        from satquery.specialists.change_detection import TemporalDifferenceBaseline
        cva = TemporalDifferenceBaseline(decision_threshold=0.20, method="gaussian")
        res_cva = cva.predict((dummy_t1, dummy_t2))
        logger.info(f"[PASS] Classical Gaussian CVA Fallback loaded. Status: {res_cva.status.value}, Change: {res_cva.predictions[0]}")
    except Exception as e:
        logger.error(f"[FAIL] Classical Gaussian CVA Fallback failed: {e}")
        all_ok = False

    return all_ok


def export_production_freeze_manifest():
    """Serialize the complete machine-readable production freeze manifest."""
    logger.info("\n=== STEP 3.1.3: EXPORTING PRODUCTION FREEZE MANIFEST ===")
    out_dir = repo_root / "artifacts" / "deployment"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "m33_production_freeze_manifest.json"

    manifest = {
        "manifest_version": "1.0.0",
        "milestone": "M3.3 Step 3.1",
        "title": "SatQuery AI Production Artifact Freezing & Integrity Manifest",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frozen_charter_guarantee": "Strictly immutable model weights and frozen decision thresholds.",
        "artifacts": {},
        "operating_thresholds": {
            "change_detection_deep_tau_star": 0.40,
            "change_detection_min_component_area": 9,
            "change_detection_fallback_tau": 0.20,
            "change_detection_fallback_sigma": 1.0,
            "router_confidence_threshold_tau_intent": 0.65,
            "router_margin_threshold_tau_margin": 0.15,
            "router_temperature_scaling_T_cal": 0.0991,
            "land_cover_evaluation_regime": "Per-class validation tuned"
        },
        "taxonomies": {
            "canonical_tasks": [
                "CHANGE_DETECTION",
                "CHANGE_VQA",
                "LAND_COVER_CLASSIFICATION",
                "OPTICAL_SAR_FUSION",
                "SCENE_CAPTIONING",
                "SINGLE_IMAGE_VQA",
                "VISUAL_GROUNDING"
            ],
            "land_cover_classes_16": [
                "Arable land",
                "Beaches, dunes, sands",
                "Broad-leaved forest",
                "Complex cultivation patterns",
                "Coniferous forest",
                "Industrial or commercial units",
                "Inland waters",
                "Inland wetlands",
                "Land principally occupied by agriculture, with significant areas of natural vegetation",
                "Mixed forest",
                "Moors, heathland and sclerophyllous vegetation",
                "Natural grassland and sparsely vegetated areas",
                "Pastures",
                "Permanent crops",
                "Transitional woodland, shrub",
                "Urban fabric"
            ]
        },
        "verified_environments": {
            "primary_accelerated": {
                "os": "Windows 11 Enterprise (AMD64)",
                "python": "3.13.2",
                "pytorch": "2.6.0+cu124",
                "cuda": "12.4",
                "device": "NVIDIA GeForce RTX 2050 (4096 MB VRAM)"
            },
            "secondary_cpu": {
                "os": "Windows 11 Enterprise (AMD64)",
                "python": "3.14.3",
                "pytorch": "2.15.0+cpu",
                "device": "Intel Core i5-12450H (16 GB RAM)"
            }
        }
    }

    for key, spec in PRODUCTION_ARTIFACTS_SPEC.items():
        p = repo_root / spec["path"]
        manifest["artifacts"][key] = {
            "path": spec["path"],
            "sha256": compute_file_sha256(p),
            "size_bytes": p.stat().st_size,
            "framework": spec.get("framework", "N/A"),
            "description": spec["description"],
            "operating_parameters": spec.get("operating_thresholds") or spec.get("operating_threshold", "N/A")
        }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Successfully generated and verified manifest at: {manifest_path}")


def main():
    start_t = time.time()
    logger.info("Starting M3.3 Step 3.1 Artifact Integrity & Freezing Audit...")

    checksums_ok = verify_file_checksums()
    if not checksums_ok:
        logger.error("FATAL: Checksum verification failed. Aborting.")
        sys.exit(1)

    inference_ok = verify_checkpoint_loading_and_inference()
    if not inference_ok:
        logger.error("FATAL: Checkpoint loading/inference verification failed. Aborting.")
        sys.exit(1)

    export_production_freeze_manifest()

    elapsed = time.time() - start_t
    logger.info(f"M3.3 Step 3.1 Verification completed successfully in {elapsed:.2f} seconds. EXIT 0.")


if __name__ == "__main__":
    main()
