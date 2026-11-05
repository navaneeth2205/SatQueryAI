"""
Orchestration Script for Milestone M3.1 Deep Land-Cover Specialist Experiments.

Executes sequential GPU training for:
1. Stage 2: Dual-Branch ResNet-34 with Class-Balanced BCE (CB-BCE, beta=0.999).
2. Stage 3: Dual-Branch ResNet-34 with Asymmetric Loss (ASL, gamma+=0, gamma-=4, m=0.05).
3. Comparative analysis against Phase 2 Classical Baseline (Macro F1 = 47.39%).

Hardware: NVIDIA GeForce RTX 2050 (4 GB VRAM)
Protocol: PHASE3_M30_PROTOCOL_FREEZE.md (Deliverable F.1)
"""

import sys
from pathlib import Path
import json
import logging
import time

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.train_deep_land_cover import run_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("m31_orchestrator")


def main():
    logger.info("=" * 80)
    logger.info("STARTING MILESTONE M3.1 FULL-DATA SPECIALIST TRAINING ON NVIDIA RTX 2050")
    logger.info("=" * 80)

    results = {}

    # -------------------------------------------------------------------------
    # STAGE 2: CB-BCE FULL-DATA TRAINING (N = 7,180)
    # -------------------------------------------------------------------------
    logger.info("LAUNCHING STAGE 2: CB-BCE Full-Data Training...")
    cb_bce_dir = repo_root / "artifacts" / "specialists" / "deep_land_cover_cb_bce"
    t0_cb = time.time()
    try:
        cb_summary = run_experiment(
            loss_type="cb_bce",
            epochs=40,
            batch_size=64,
            learning_rate=1e-4,
            patience=7,
            seed=42,
            output_dir=cb_bce_dir
        )
        duration_cb = time.time() - t0_cb
        results["cb_bce"] = cb_summary
        logger.info(f"STAGE 2 [CB-BCE] COMPLETED IN {duration_cb / 60.0:.1f} MINUTES.")
        logger.info(
            f"CB-BCE Test Macro F1 (Tuned): {cb_summary['test_metrics_tuned']['macro_f1_16']:.4f} "
            f"(Present: {cb_summary['test_metrics_tuned']['macro_f1_present']:.4f}, Micro: {cb_summary['test_metrics_tuned']['micro_f1']:.4f})"
        )
    except Exception as e:
        logger.error(f"STAGE 2 [CB-BCE] FAILED WITH ERROR: {e}", exc_info=True)
        raise

    # -------------------------------------------------------------------------
    # STAGE 3: ASL FULL-DATA TRAINING (N = 7,180)
    # -------------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("LAUNCHING STAGE 3: ASL Full-Data Training...")
    asl_dir = repo_root / "artifacts" / "specialists" / "deep_land_cover_asl"
    t0_asl = time.time()
    try:
        asl_summary = run_experiment(
            loss_type="asl",
            epochs=40,
            batch_size=64,
            learning_rate=1e-4,
            patience=7,
            seed=42,
            output_dir=asl_dir
        )
        duration_asl = time.time() - t0_asl
        results["asl"] = asl_summary
        logger.info(f"STAGE 3 [ASL] COMPLETED IN {duration_asl / 60.0:.1f} MINUTES.")
        logger.info(
            f"ASL Test Macro F1 (Tuned): {asl_summary['test_metrics_tuned']['macro_f1_16']:.4f} "
            f"(Present: {asl_summary['test_metrics_tuned']['macro_f1_present']:.4f}, Micro: {asl_summary['test_metrics_tuned']['micro_f1']:.4f})"
        )
    except Exception as e:
        logger.error(f"STAGE 3 [ASL] FAILED WITH ERROR: {e}", exc_info=True)
        raise

    # -------------------------------------------------------------------------
    # COMPARATIVE SUMMARY MANIFEST
    # -------------------------------------------------------------------------
    comp_manifest_path = repo_root / "artifacts" / "specialists" / "m31_model_comparison.json"
    comp_manifest = {
        "hardware": "NVIDIA GeForce RTX 2050 (4 GB VRAM)",
        "train_samples": 7180,
        "val_samples": 3255,
        "test_samples": 3248,
        "classical_baseline_macro_f1": 0.4739,
        "cb_bce": {
            "best_epoch": cb_summary["best_epoch"],
            "training_duration_sec": cb_summary["total_training_duration_sec"],
            "val_macro_f1_tuned": cb_summary["validation_metrics_tuned"]["macro_f1_16"],
            "test_macro_f1_default": cb_summary["test_metrics_default"]["macro_f1_16"],
            "test_macro_f1_tuned": cb_summary["test_metrics_tuned"]["macro_f1_16"],
            "test_macro_f1_present": cb_summary["test_metrics_tuned"]["macro_f1_present"],
            "test_micro_f1": cb_summary["test_metrics_tuned"]["micro_f1"],
            "test_hamming_loss": cb_summary["test_metrics_tuned"]["hamming_loss"],
        },
        "asl": {
            "best_epoch": asl_summary["best_epoch"],
            "training_duration_sec": asl_summary["total_training_duration_sec"],
            "val_macro_f1_tuned": asl_summary["validation_metrics_tuned"]["macro_f1_16"],
            "test_macro_f1_default": asl_summary["test_metrics_default"]["macro_f1_16"],
            "test_macro_f1_tuned": asl_summary["test_metrics_tuned"]["macro_f1_16"],
            "test_macro_f1_present": asl_summary["test_metrics_tuned"]["macro_f1_present"],
            "test_micro_f1": asl_summary["test_metrics_tuned"]["micro_f1"],
            "test_hamming_loss": asl_summary["test_metrics_tuned"]["hamming_loss"],
        }
    }

    with open(comp_manifest_path, "w", encoding="utf-8") as f:
        json.dump(comp_manifest, f, indent=2)

    logger.info("=" * 80)
    logger.info("MILESTONE M3.1 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    logger.info(f"Comparison manifest written to: {comp_manifest_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
