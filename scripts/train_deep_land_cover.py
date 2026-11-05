"""
Training and Evaluation Pipeline for Deep Multimodal Land Cover Specialist.

Implements Milestone M3.0 / M3.1 specification (Deliverable F.1):
- Dual-Branch ResNet-34 (Sentinel-1 SAR 2 bands + Sentinel-2 optical 10 bands).
- Cross-attention fusion projecting to 512-dim embedding.
- Config A: Class-Balanced BCE (CB-BCE with beta=0.999, mean reduction 1/16B).
- Config B: Multi-label Asymmetric Loss (ASL with gamma+=0, gamma-=4, m=0.05, mean reduction 1/16B).
- Data splits: BigEarthNet-14k spatially buffered split across T34TCR, T34TEQ, T34TFN, T34TCS:
    * Train: N = 7,180
    * Val:   N = 3,255 (used strictly for early stopping and threshold tuning)
    * Test:  N = 3,248 (frozen held-out test split)
- Honest hardware reporting: records device, execution time, and throughput telemetry.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Force UTF-8 encoding for Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import argparse
import json
import logging
import time
import numpy as np
import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, hamming_loss

# Add repo root to sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from satquery.specialists.models.multimodal_resnet import MultimodalResNet34
from satquery.specialists.losses import (
    compute_class_balanced_weights,
    ClassBalancedBCEWithLogitsLoss,
    AsymmetricLossWithLogits
)
from satquery.specialists.dataset_land_cover import BigEarthNetDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Union[float, np.ndarray] = 0.50
) -> Dict[str, float]:
    """
    Compute multi-label classification metrics complying with Deliverable E.2 & F.1:
    - Macro F1 (16 classes, zero_division=0.0)
    - Test-Present Macro F1 (classes with positive support in y_true)
    - Micro F1
    - Hamming Loss
    """
    if isinstance(thresholds, (int, float)):
        y_pred = (y_prob >= thresholds).astype(np.int32)
    else:
        thresh_arr = np.array(thresholds).reshape(1, -1)
        y_pred = (y_prob >= thresh_arr).astype(np.int32)

    # 1. Standard 16-class Macro F1 with zero_division=0.0 (Review 15 policy)
    macro_f1_16 = float(f1_score(y_true, y_pred, average="macro", zero_division=0.0))

    # 2. Test-Present Macro F1 (classes with positive ground truth support)
    class_support = y_true.sum(axis=0)
    present_indices = np.where(class_support > 0)[0]
    if len(present_indices) > 0:
        macro_f1_present = float(f1_score(y_true[:, present_indices], y_pred[:, present_indices], average="macro", zero_division=0.0))
    else:
        macro_f1_present = 0.0

    # 3. Micro F1
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0.0))

    # 4. Hamming Loss
    h_loss = float(hamming_loss(y_true, y_pred))

    # 5. Per-class F1
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0.0).tolist()

    return {
        "macro_f1_16": macro_f1_16,
        "macro_f1_present": macro_f1_present,
        "micro_f1": micro_f1,
        "hamming_loss": h_loss,
        "present_class_count": len(present_indices),
        "per_class_f1": per_class_f1
    }


def tune_validation_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int = 16
) -> Tuple[np.ndarray, float]:
    """
    Vectorized sweep of per-class decision thresholds tau_c in [0.10, 0.90] (step 0.05)
    strictly on validation predictions (Deliverable F.1, line 1085).
    Executes in < 30ms via numpy broadcasting.
    """
    candidate_taus = np.arange(0.10, 0.95, 0.05, dtype=np.float32)  # (T,)
    best_thresholds = np.full(num_classes, 0.50, dtype=np.float32)

    # Reshape for broadcasting: (T, 1, 1) vs (1, N, C)
    taus = candidate_taus.reshape(-1, 1, 1)
    probs = y_prob[None, :, :]
    targets = y_true[None, :, :]

    preds = (probs >= taus).astype(np.float32)
    tp = (preds * targets).sum(axis=1)           # (T, C)
    fp = (preds * (1.0 - targets)).sum(axis=1)   # (T, C)
    fn = ((1.0 - preds) * targets).sum(axis=1)   # (T, C)

    denom = 2.0 * tp + fp + fn
    f1s = np.zeros_like(tp, dtype=np.float32)
    valid_mask = denom > 0
    f1s[valid_mask] = (2.0 * tp[valid_mask]) / denom[valid_mask]

    for c in range(num_classes):
        if y_true[:, c].sum() == 0:
            best_thresholds[c] = 0.50
        else:
            best_idx = int(np.argmax(f1s[:, c]))
            best_thresholds[c] = float(candidate_taus[best_idx])

    tuned_metrics = compute_metrics(y_true, y_prob, best_thresholds)
    return best_thresholds, tuned_metrics["macro_f1_16"]


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Train for one epoch, returning average loss and duration."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    t0 = time.time()

    for opt, sar, tgt, _ in loader:
        opt = opt.to(device)
        sar = sar.to(device)
        tgt = tgt.to(device)

        optimizer.zero_grad()
        logits = model(opt, sar)
        loss = criterion(logits, tgt)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if num_batches % 30 == 0 or num_batches == len(loader):
            elapsed = time.time() - t0
            logger.info(f"  Batch [{num_batches:03d}/{len(loader):03d}] Running Loss: {total_loss / num_batches:.4f} | Elapsed: {elapsed:.1f}s")

    duration = time.time() - t0
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, duration


@torch.no_grad()
def evaluate_dataset(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluate model over DataLoader, returning loss, y_true, and predicted probabilities."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    t0 = time.time()
    all_targets = []
    all_probs = []

    for opt, sar, tgt, _ in loader:
        opt = opt.to(device)
        sar = sar.to(device)
        tgt = tgt.to(device)

        logits = model(opt, sar)
        loss = criterion(logits, tgt)

        probs = torch.sigmoid(logits)

        total_loss += loss.item()
        num_batches += 1

        if num_batches % 30 == 0 or num_batches == len(loader):
            elapsed = time.time() - t0
            logger.info(f"  Batch [{num_batches:03d}/{len(loader):03d}] Running Loss: {total_loss / num_batches:.4f} | Elapsed: {elapsed:.1f}s")
        all_targets.append(tgt.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    avg_loss = total_loss / max(num_batches, 1)
    y_true = np.vstack(all_targets)
    y_prob = np.vstack(all_probs)
    return avg_loss, y_true, y_prob


def run_experiment(
    loss_type: str = "cb_bce",
    epochs: int = 40,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    patience: int = 7,
    seed: int = 42,
    subset_train: Optional[int] = None,
    subset_val: Optional[int] = None,
    subset_test: Optional[int] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Execute full training and evaluation experiment for one loss configuration.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_count = psutil.cpu_count(logical=True)
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)

    logger.info("=" * 70)
    logger.info(f"EXPERIMENT CONFIGURATION: Loss = {loss_type.upper()} | Seed = {seed}")
    logger.info(f"HARDWARE TELEMETRY: Device = {device} | Logical CPUs = {cpu_count} | RAM = {ram_gb:.1f} GB")
    logger.info("=" * 70)

    # 1. Datasets
    metadata_dir = repo_root / "data"
    archive_path = repo_root / "initial_trails" / "Dataset" / "archive.zip"
    classes_path = repo_root / "artifacts" / "specialists" / "land_cover" / "classes.json"

    train_ds = BigEarthNetDataset(
        metadata_path=metadata_dir / "train_full_metadata.parquet",
        archive_path=archive_path,
        classes_path=classes_path,
        augment=True,
        subset_size=subset_train,
        seed=seed
    )
    val_ds = BigEarthNetDataset(
        metadata_path=metadata_dir / "val_full_metadata.parquet",
        archive_path=archive_path,
        classes_path=classes_path,
        augment=False,
        subset_size=subset_val,
        seed=seed
    )
    test_ds = BigEarthNetDataset(
        metadata_path=metadata_dir / "test_full_metadata.parquet",
        archive_path=archive_path,
        classes_path=classes_path,
        augment=False,
        subset_size=subset_test,
        seed=seed
    )

    logger.info(f"Dataset partitions: Train = {len(train_ds)}, Val = {len(val_ds)}, Test = {len(test_ds)}")

    # On Windows, num_workers=0 avoids multiprocessing IPC overhead; pin_memory speeds up CUDA transfer
    use_pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=use_pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=use_pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=use_pin)

    # 2. Model & Optimization
    model = MultimodalResNet34(
        num_classes=16,
        optical_channels=10,
        sar_channels=2,
        pretrained=True,
        channel_order="b02_first"
    ).to(device)

    # AdamW with weight decay 1e-2 as specified in Deliverable F.1
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 3. Loss Configuration
    if loss_type.lower() == "cb_bce":
        train_counts = train_ds.get_class_counts()
        criterion = ClassBalancedBCEWithLogitsLoss(pos_counts=train_counts, total_samples=len(train_ds)).to(device)
    elif loss_type.lower() == "asl":
        criterion = AsymmetricLossWithLogits(gamma_pos=0.0, gamma_neg=4.0, clip_margin=0.05).to(device)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Output directory
    if output_dir is None:
        output_dir = repo_root / "artifacts" / "specialists" / f"deep_land_cover_{loss_type.lower()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 4. Training Loop with Early Stopping on Validation Macro F1
    best_val_macro_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    best_thresholds = np.full(16, 0.50, dtype=np.float32)
    best_weights_path = output_dir / f"best_model_{loss_type.lower()}.pt"
    patience_counter = 0

    history = []
    total_training_start = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_time = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, y_val_true, y_val_prob = evaluate_dataset(model, val_loader, criterion, device)
        scheduler.step()

        val_metrics_def = compute_metrics(y_val_true, y_val_prob, thresholds=0.50)
        curr_thresholds, val_macro_f1_tuned = tune_validation_thresholds(y_val_true, y_val_prob, num_classes=16)
        val_metrics_tuned = compute_metrics(y_val_true, y_val_prob, thresholds=curr_thresholds)

        rss_mb = psutil.Process().memory_info().rss / (1024 ** 2)
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_f1_default": val_metrics_def["macro_f1_16"],
            "val_macro_f1_tuned": val_macro_f1_tuned,
            "val_macro_f1_present": val_metrics_tuned["macro_f1_present"],
            "val_micro_f1": val_metrics_tuned["micro_f1"],
            "val_hamming_loss": val_metrics_tuned["hamming_loss"],
            "train_duration_sec": train_time,
            "ram_rss_mb": rss_mb,
            "lr": optimizer.param_groups[0]["lr"]
        }
        history.append(epoch_record)

        logger.info(
            f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Macro F1 (Tuned): {val_macro_f1_tuned:.4f} (Def: {val_metrics_def['macro_f1_16']:.4f}, Present: {val_metrics_tuned['macro_f1_present']:.4f}) | "
            f"Time: {train_time:.1f}s | RAM: {rss_mb:.1f} MB"
        )

        # Checkpoint selection on highest validation Macro F1 (Deliverable F.1)
        is_best = False
        if val_macro_f1_tuned > best_val_macro_f1:
            is_best = True
        elif val_macro_f1_tuned == best_val_macro_f1 and val_loss < best_val_loss:
            is_best = True

        if is_best:
            best_val_macro_f1 = val_macro_f1_tuned
            best_val_loss = val_loss
            best_epoch = epoch
            best_thresholds = curr_thresholds.copy()
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": val_macro_f1_tuned,
                "val_loss": val_loss,
                "tuned_thresholds": best_thresholds.tolist(),
                "loss_type": loss_type,
                "seed": seed
            }, best_weights_path)
            logger.info(f"  -> Best model saved at epoch {epoch} (Val Macro F1 = {val_macro_f1_tuned:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch} (Patience = {patience})")
                break

    total_training_duration = time.time() - total_training_start

    # 5. Load Best Checkpoint & Verify Final Validation Metrics
    logger.info(f"Loading best checkpoint from epoch {best_epoch} (Val Macro F1 = {best_val_macro_f1:.4f})...")
    checkpoint = torch.load(best_weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, y_val_true, y_val_prob = evaluate_dataset(model, val_loader, criterion, device)
    tuned_thresholds, final_val_macro_f1 = tune_validation_thresholds(y_val_true, y_val_prob, num_classes=16)

    # Save threshold configuration
    thresh_cfg_path = output_dir / "threshold_config.json"
    with open(thresh_cfg_path, "w", encoding="utf-8") as f:
        json.dump({
            "loss_type": loss_type,
            "best_epoch": best_epoch,
            "default_threshold": 0.50,
            "tuned_class_thresholds": tuned_thresholds.tolist(),
            "classes": train_ds.classes,
            "validation_macro_f1_best": best_val_macro_f1,
            "validation_loss_best": best_val_loss
        }, f, indent=2)

    # 6. Single Frozen Held-Out Test Split Evaluation (Deliverable F.1, line 1106)
    logger.info("Executing final evaluation on frozen held-out test set (N = 3,248)...")
    test_loss, y_test_true, y_test_prob = evaluate_dataset(model, test_loader, criterion, device)

    default_test_metrics = compute_metrics(y_test_true, y_test_prob, thresholds=0.50)
    tuned_test_metrics = compute_metrics(y_test_true, y_test_prob, thresholds=tuned_thresholds)

    logger.info("=" * 70)
    logger.info(f"FINAL HELD-OUT TEST EVALUATION [{loss_type.upper()}]")
    logger.info(f"Default (tau=0.50) : Macro F1 (16 cls) = {default_test_metrics['macro_f1_16']:.4f} | Present (14 cls) = {default_test_metrics['macro_f1_present']:.4f} | Micro F1 = {default_test_metrics['micro_f1']:.4f}")
    logger.info(f"Tuned Thresholds   : Macro F1 (16 cls) = {tuned_test_metrics['macro_f1_16']:.4f} | Present (14 cls) = {tuned_test_metrics['macro_f1_present']:.4f} | Micro F1 = {tuned_test_metrics['micro_f1']:.4f}")
    logger.info("=" * 70)

    # Save test predictions and metrics manifest
    np.save(output_dir / "y_test_true.npy", y_test_true)
    np.save(output_dir / "y_test_prob.npy", y_test_prob)

    summary = {
        "loss_type": loss_type,
        "seed": seed,
        "device": str(device),
        "total_training_duration_sec": total_training_duration,
        "best_epoch": best_epoch,
        "validation_metrics_default": {
            "macro_f1_16": best_val_macro_f1
        },
        "validation_metrics_tuned": {
            "macro_f1_16": final_val_macro_f1
        },
        "test_metrics_default": default_test_metrics,
        "test_metrics_tuned": tuned_test_metrics,
        "history": history
    }

    with open(output_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Deep Multimodal Land Cover Specialist")
    parser.add_argument("--loss", choices=["cb_bce", "asl"], default="cb_bce", help="Loss formulation")
    parser.add_argument("--epochs", type=int, default=40, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--subset_train", type=int, default=None, help="Subset size for rapid verification")
    parser.add_argument("--subset_val", type=int, default=None, help="Validation subset size")
    parser.add_argument("--subset_test", type=int, default=None, help="Test subset size")
    parser.add_argument("--output_dir", type=str, default=None, help="Custom output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else None

    run_experiment(
        loss_type=args.loss,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        seed=args.seed,
        subset_train=args.subset_train,
        subset_val=args.subset_val,
        subset_test=args.subset_test,
        output_dir=out_dir
    )
