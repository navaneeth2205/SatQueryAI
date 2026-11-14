"""
Milestone M3.2 - Step 3: Controlled Ablation Sequence & Specialist Architecture Training.
Document Code: SATQUERY-SPECIALIST-M3.2-STEP3-TRAIN

Executes the step-by-step ablation study authorized by the reviewer:
E1: ResNet-18 Siamese U-Net (scratch, pure difference, no attention)
E2: E1 + ImageNet Pretrained Encoder
E3: E2 + Richer Temporal Fusion (Concat + Difference with 1x1 projection)
E4: E3 + scSE Spatial-Channel Squeeze & Excitation Attention
E5: Loss Function Ablation on Winning Architecture (BCE+Dice vs Focal vs Focal-Tversky)

Evaluation Protocol:
- Strictly disjoint 10 Train scenes, 4 Validation scenes, 10 Held-out Test scenes
- Model selection and threshold freezing strictly on 4 validation scenes
- Test set evaluated with frozen parameters
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from satquery.specialists.dataset_change_detection import (
    OSCDPatchDataset,
    OSCDFullScene,
)
from satquery.specialists.models.siamese_resnet_unet import (
    SiameseResNet18UNet,
)
from satquery.specialists.losses_change_detection import (
    BCEDiceLoss,
    BinaryFocalLoss,
    FocalTverskyLoss,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("Step3Ablation")

DATA_DIR = REPO_ROOT / "data" / "oscd"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "specialists" / "change_detection_step3"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes standard binary segmentation metrics."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    iou = (tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "iou": round(iou, 4),
    }


def evaluate_scenes_at_threshold(
    model: nn.Module,
    scene_containers: List[OSCDFullScene],
    device: torch.device,
    threshold: float,
) -> Tuple[Dict[str, Dict], Dict[str, float]]:
    """Evaluates full scene inference across containers at a specific decision threshold."""
    per_scene = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    f1_list = []

    for sc in scene_containers:
        prob_map = sc.predict_full_scene(model, device)
        pred_mask = (prob_map >= threshold).astype(np.uint8)
        m = calculate_binary_metrics(sc.cm_true, pred_mask)

        per_scene[sc.scene_name] = m
        total_tp += m["tp"]
        total_fp += m["fp"]
        total_fn += m["fn"]
        total_tn += m["tn"]
        f1_list.append(m["f1"])

    pooled_prec = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
    pooled_rec = (total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
    pooled_f1 = (2 * total_tp / (2 * total_tp + total_fp + total_fn)) if (2 * total_tp + total_fp + total_fn) > 0 else 0.0
    pooled_iou = (total_tp / (total_tp + total_fp + total_fn)) if (total_tp + total_fp + total_fn) > 0 else 0.0

    summary = {
        "macro_f1": round(float(np.mean(f1_list)), 4),
        "pooled_f1": round(pooled_f1, 4),
        "pooled_precision": round(pooled_prec, 4),
        "pooled_recall": round(pooled_rec, 4),
        "pooled_iou": round(pooled_iou, 4),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_tn": total_tn,
    }
    return per_scene, summary


def sweep_validation_threshold(
    model: nn.Module,
    val_scenes: List[OSCDFullScene],
    device: torch.device,
    thresholds: Optional[List[float]] = None,
) -> Tuple[float, float, Dict]:
    """Sweeps threshold strictly on validation scenes to find optimal Macro F1 using cached probability maps."""
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.10, 0.95, 0.05)]

    # Precompute probability maps once for all validation scenes
    cached_prob_maps = []
    for sc in val_scenes:
        prob_map = sc.predict_full_scene(model, device)
        cached_prob_maps.append((sc, prob_map))

    best_tau = 0.50
    best_val_macro_f1 = -1.0
    best_summary = None

    for tau in thresholds:
        per_scene = {}
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_tn = 0
        f1_list = []

        for sc, prob_map in cached_prob_maps:
            pred_mask = (prob_map >= tau).astype(np.uint8)
            m = calculate_binary_metrics(sc.cm_true, pred_mask)
            per_scene[sc.scene_name] = m
            total_tp += m["tp"]
            total_fp += m["fp"]
            total_fn += m["fn"]
            total_tn += m["tn"]
            f1_list.append(m["f1"])

        pooled_prec = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
        pooled_rec = (total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
        pooled_f1 = (2 * total_tp / (2 * total_tp + total_fp + total_fn)) if (2 * total_tp + total_fp + total_fn) > 0 else 0.0
        pooled_iou = (total_tp / (total_tp + total_fp + total_fn)) if (total_tp + total_fp + total_fn) > 0 else 0.0

        macro_f1 = round(float(np.mean(f1_list)), 4)
        if macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = macro_f1
            best_tau = tau
            best_summary = {
                "macro_f1": macro_f1,
                "pooled_f1": round(pooled_f1, 4),
                "pooled_precision": round(pooled_prec, 4),
                "pooled_recall": round(pooled_rec, 4),
                "pooled_iou": round(pooled_iou, 4),
                "total_tp": total_tp,
                "total_fp": total_fp,
                "total_fn": total_fn,
                "total_tn": total_tn,
            }

    logger.info(f"Optimal Validation Threshold: tau* = {best_tau:.2f} (Val Macro F1 = {best_val_macro_f1*100:.2f}%)")
    return best_tau, best_val_macro_f1, best_summary


def train_single_experiment(
    exp_id: str,
    exp_desc: str,
    model_kwargs: Dict,
    loss_module: nn.Module,
    train_loader: DataLoader,
    val_scenes: List[OSCDFullScene],
    device: torch.device,
    epochs: int = 20,
    lr: float = 3e-4,
) -> Dict:
    """Trains a single ablation experiment, checkpointing peak validation performance."""
    logger.info("=" * 80)
    logger.info(f"RUNNING EXPERIMENT {exp_id}: {exp_desc}")
    logger.info(f"Configuration: {model_kwargs} | Loss: {loss_module.__class__.__name__}")
    logger.info("=" * 80)

    seed_everything(42)

    model = SiameseResNet18UNet(**model_kwargs).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model instantiated. Total Trainable Parameters: {param_count:,d}")

    # Differential learning rate if pretrained
    if model_kwargs.get("pretrained", False):
        stem_layer_params = [
            p for n, p in model.named_parameters()
            if any(k in n for k in ["stem", "layer1", "layer2", "layer3", "layer4"])
        ]
        decoder_params = [
            p for n, p in model.named_parameters()
            if not any(k in n for k in ["stem", "layer1", "layer2", "layer3", "layer4"])
        ]
        optimizer = torch.optim.AdamW([
            {"params": stem_layer_params, "lr": lr * 0.3},
            {"params": decoder_params, "lr": lr},
        ], weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_macro_f1 = -1.0
    best_epoch = -1
    ckpt_path = ARTIFACTS_DIR / f"{exp_id.lower()}_best_model.pt"

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        batch_count = 0

        for t1, t2, cm in train_loader:
            t1, t2, cm = t1.to(device), t2.to(device), cm.to(device)

            optimizer.zero_grad()
            logits = model(t1, t2)
            loss = loss_module(logits, cm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item()
            batch_count += 1

        scheduler.step()
        train_loss /= max(batch_count, 1)

        # Quick validation evaluation at fixed tau = 0.50
        _, val_metrics = evaluate_scenes_at_threshold(model, val_scenes, device, threshold=0.50)
        val_macro_f1 = val_metrics["macro_f1"]

        logger.info(
            f"[{exp_id}] Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | "
            f"Val Macro F1 (tau=0.50): {val_macro_f1*100:.2f}% | Pooled F1: {val_metrics['pooled_f1']*100:.2f}%"
        )

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path)

    elapsed_time = time.time() - start_time
    logger.info(
        f"[{exp_id}] Training complete in {elapsed_time:.1f}s ({elapsed_time/60:.2f} min). "
        f"Best Epoch: {best_epoch} with Val Macro F1: {best_val_macro_f1*100:.2f}%"
    )

    # Load best checkpoint for validation threshold sweep
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    best_tau, optimal_val_macro_f1, val_summary = sweep_validation_threshold(model, val_scenes, device)

    return {
        "exp_id": exp_id,
        "description": exp_desc,
        "parameters": param_count,
        "model_kwargs": model_kwargs,
        "loss_name": loss_module.__class__.__name__,
        "training_time_sec": round(elapsed_time, 2),
        "best_epoch": best_epoch,
        "fixed_val_macro_f1": best_val_macro_f1,
        "optimal_tau": best_tau,
        "optimal_val_macro_f1": optimal_val_macro_f1,
        "val_summary": val_summary,
        "checkpoint_path": str(ckpt_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Run M3.2 Step 3 Ablation Experiments")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs per experiment")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Target Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load datasets
    logger.info("Initializing OSCD dataset loaders...")
    train_dataset = OSCDPatchDataset(DATA_DIR, "train", patch_size=256, stride=64, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    val_dirs = sorted([p for p in (DATA_DIR / "val").iterdir() if p.is_dir()])
    val_scenes = [OSCDFullScene(p) for p in val_dirs]

    test_dirs = sorted([p for p in (DATA_DIR / "test").iterdir() if p.is_dir()])
    test_scenes = [OSCDFullScene(p) for p in test_dirs]

    logger.info(f"Loaded {len(train_dataset)} train patches, {len(val_scenes)} val scenes, {len(test_scenes)} test scenes.")

    # -------------------------------------------------------------
    # PHASE A: Architecture Ablation Sequence (E1, E2, E3, E4)
    # Loss fixed to BCEDiceLoss(pos_weight=2.0)
    # -------------------------------------------------------------
    base_loss = BCEDiceLoss(dice_weight=1.0, pos_weight=2.0)

    ablation_matrix = [
        (
            "E1",
            "ResNet-18 Siamese U-Net (Scratch, Diff, No scSE)",
            {"in_channels": 3, "num_classes": 1, "pretrained": False, "fusion_mode": "diff", "use_scse": False},
        ),
        (
            "E2",
            "E1 + ImageNet Pretrained Encoder (Transfer Learning)",
            {"in_channels": 3, "num_classes": 1, "pretrained": True, "fusion_mode": "diff", "use_scse": False},
        ),
        (
            "E3",
            "E2 + Richer Temporal Fusion (Diff + Concat with 1x1 proj)",
            {"in_channels": 3, "num_classes": 1, "pretrained": True, "fusion_mode": "diff_cat", "use_scse": False},
        ),
        (
            "E4",
            "E3 + scSE Spatial-Channel Squeeze & Excitation Attention",
            {"in_channels": 3, "num_classes": 1, "pretrained": True, "fusion_mode": "diff_cat", "use_scse": True},
        ),
    ]

    ablation_results = []
    for exp_id, exp_desc, model_kwargs in ablation_matrix:
        res = train_single_experiment(
            exp_id=exp_id,
            exp_desc=exp_desc,
            model_kwargs=model_kwargs,
            loss_module=base_loss,
            train_loader=train_loader,
            val_scenes=val_scenes,
            device=device,
            epochs=args.epochs,
        )
        ablation_results.append(res)

    # Determine winning architecture strictly on validation Macro F1
    best_arch_exp = max(ablation_results, key=lambda r: r["optimal_val_macro_f1"])
    logger.info("=" * 80)
    logger.info(f"PHASE A COMPLETE: Best Architecture based strictly on Validation Macro F1 is {best_arch_exp['exp_id']}")
    logger.info(f"  Description: {best_arch_exp['description']}")
    logger.info(f"  Validation Macro F1: {best_arch_exp['optimal_val_macro_f1']*100:.2f}% (tau*={best_arch_exp['optimal_tau']})")
    logger.info("=" * 80)

    # -------------------------------------------------------------
    # PHASE B: Loss Function Ablation (E5) on Winning Architecture
    # -------------------------------------------------------------
    winning_kwargs = best_arch_exp["model_kwargs"]
    loss_experiments = [
        ("E5_BCE_Dice", "Best Architecture + BCEDiceLoss (Baseline)", BCEDiceLoss(dice_weight=1.0, pos_weight=2.0)),
        ("E5_Focal", "Best Architecture + BinaryFocalLoss (alpha=0.25, gamma=2.0)", BinaryFocalLoss(alpha=0.25, gamma=2.0)),
        ("E5_FocalTversky", "Best Architecture + FocalTverskyLoss (alpha=0.7, beta=0.3, gamma=1.33)", FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=1.333)),
    ]

    loss_results = []
    for exp_id, exp_desc, loss_mod in loss_experiments:
        if exp_id == "E5_BCE_Dice":
            # Already trained as best_arch_exp
            res = dict(best_arch_exp)
            res["exp_id"] = exp_id
            res["description"] = exp_desc
            loss_results.append(res)
        else:
            res = train_single_experiment(
                exp_id=exp_id,
                exp_desc=exp_desc,
                model_kwargs=winning_kwargs,
                loss_module=loss_mod,
                train_loader=train_loader,
                val_scenes=val_scenes,
                device=device,
                epochs=args.epochs,
            )
            loss_results.append(res)

    # Determine overall champion strictly on validation Macro F1
    champion_exp = max(loss_results, key=lambda r: r["optimal_val_macro_f1"])
    logger.info("=" * 80)
    logger.info(f"PHASE B COMPLETE: Selected Champion Specialist: {champion_exp['exp_id']}")
    logger.info(f"  Description: {champion_exp['description']}")
    logger.info(f"  Frozen Optimal Threshold: tau* = {champion_exp['optimal_tau']}")
    logger.info(f"  Peak Validation Macro F1: {champion_exp['optimal_val_macro_f1']*100:.2f}%")
    logger.info("=" * 80)

    # -------------------------------------------------------------
    # PHASE C: Unblinding Held-Out Test Set STRICTLY ONCE
    # -------------------------------------------------------------
    logger.info("PHASE C: Unblinding Held-Out Test Set (10 Scenes) Strictly Once for Champion Evaluation...")
    champ_model = SiameseResNet18UNet(**champion_exp["model_kwargs"]).to(device)
    champ_model.load_state_dict(torch.load(champion_exp["checkpoint_path"], map_location=device))
    champ_model.eval()

    test_per_scene, test_summary = evaluate_scenes_at_threshold(
        champ_model, test_scenes, device, threshold=champion_exp["optimal_tau"]
    )

    logger.info("=" * 80)
    logger.info(f"CHAMPION EVALUATION ON HELD-OUT TEST SPLIT (tau*={champion_exp['optimal_tau']}):")
    logger.info(f"  Test Macro F1          : {test_summary['macro_f1']*100:.2f}%")
    logger.info(f"  Test Pooled F1         : {test_summary['pooled_f1']*100:.2f}%")
    logger.info(f"  Test Pooled Precision  : {test_summary['pooled_precision']*100:.2f}%")
    logger.info(f"  Test Pooled Recall     : {test_summary['pooled_recall']*100:.2f}%")
    logger.info(f"  Test Pooled IoU        : {test_summary['pooled_iou']*100:.2f}%")
    logger.info("=" * 80)

    # Count passing scenes (F1 >= 0.50)
    passing_scenes = [sc for sc, m in test_per_scene.items() if m["f1"] >= 0.50]
    logger.info(f"Passing Scenes (F1 >= 0.50): {len(passing_scenes)} / 10 ({passing_scenes})")

    # Serialize complete results ledger
    full_audit_ledger = {
        "milestone": "M3.2",
        "step": "Step 3: Siamese ResNet-18 U-Net Development & Ablation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "architecture_ablation": ablation_results,
        "loss_ablation": loss_results,
        "selected_champion": {
            "exp_id": champion_exp["exp_id"],
            "description": champion_exp["description"],
            "parameters": champion_exp["parameters"],
            "frozen_tau": champion_exp["optimal_tau"],
            "val_macro_f1": champion_exp["optimal_val_macro_f1"],
            "test_summary": test_summary,
            "test_per_scene": test_per_scene,
            "passing_scenes_count": len(passing_scenes),
            "passing_scenes": passing_scenes,
            "target_met": (len(passing_scenes) >= 5 and test_summary["macro_f1"] >= 0.50),
        },
    }

    ledger_path = ARTIFACTS_DIR / "step3_ablation_audit_ledger.json"
    with open(ledger_path, "w", encoding="utf-8") as f:
        json.dump(full_audit_ledger, f, indent=2)

    logger.info(f"Saved complete Step 3 audit ledger to: {ledger_path}")


if __name__ == "__main__":
    main()
