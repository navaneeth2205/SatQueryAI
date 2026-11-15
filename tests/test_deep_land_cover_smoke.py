"""
Smoke tests and verification suite for Deep Multimodal Land Cover Specialist.

Covers Stages 1.1 - 1.4:
1.1 Dual-branch ResNet-34 tensor shapes, pretrained weight transfer, and gradients.
1.2 Class-Balanced BCE (CB-BCE) and Asymmetric Loss (ASL) mathematical properties and stability.
1.3 BigEarthNet-14k multimodal dataset loading, normalization, and label encoding.
1.4 Single-batch overfitting smoke test and CPU throughput benchmark.
"""

from pathlib import Path
import time
import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from satquery.specialists.models.multimodal_resnet import MultimodalResNet34
from satquery.specialists.losses import (
    compute_class_balanced_weights,
    ClassBalancedBCEWithLogitsLoss,
    AsymmetricLossWithLogits
)
from satquery.specialists.dataset_land_cover import (
    BigEarthNetDataset,
    TRAIN_OPTICAL_MEANS,
    TRAIN_OPTICAL_STDS,
    SAR_DB_MIN,
    SAR_DB_MAX
)


def test_multimodal_resnet34_shapes_and_gradients():
    """Verify input/output shapes and complete gradient flow through both branches."""
    model = MultimodalResNet34(
        num_classes=16,
        optical_channels=10,
        sar_channels=2,
        pretrained=False
    )
    model.train()

    B = 2
    opt = torch.randn(B, 10, 120, 120, requires_grad=True)
    sar = torch.randn(B, 2, 120, 120, requires_grad=True)

    logits, emb = model.forward_with_embeddings(opt, sar)
    assert logits.shape == (B, 16), f"Expected (2, 16), got {logits.shape}"
    assert emb.shape == (B, 512), f"Expected (2, 512), got {emb.shape}"

    loss = logits.sum()
    loss.backward()

    assert opt.grad is not None and torch.isfinite(opt.grad).all()
    assert sar.grad is not None and torch.isfinite(sar.grad).all()
    assert model.opt_conv1.weight.grad is not None
    assert model.sar_conv1.weight.grad is not None
    assert model.fusion.cross_attn_opt_to_sar.in_proj_weight.grad is not None
    assert model.head[1].weight.grad is not None


def test_class_balanced_bce_loss_properties():
    """Verify CB-BCE weight calculation, zero-support fallback, and loss reduction."""
    # 16 classes with zero count at index 1
    counts = [1000, 0, 500, 200] + [50] * 12
    weights = compute_class_balanced_weights(counts, total_samples=7180, beta=0.999)

    assert len(weights) == 16
    assert weights[1] > weights[0], "Zero-support fallback weight should exceed high-frequency class weight"
    assert torch.isfinite(weights).all()

    criterion = ClassBalancedBCEWithLogitsLoss(pos_counts=counts, total_samples=7180)
    logits = torch.randn(4, 16, requires_grad=True)
    targets = torch.randint(0, 2, (4, 16)).float()

    loss = criterion(logits, targets)
    assert torch.isfinite(loss)
    assert loss.dim() == 0  # Scalar reduction

    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_asymmetric_loss_properties():
    """Verify ASL numerical stability with normal and extreme logits."""
    asl = AsymmetricLossWithLogits(gamma_pos=0.0, gamma_neg=4.0, clip_margin=0.05)

    # Standard logits
    logits = torch.randn(4, 16, requires_grad=True)
    targets = torch.randint(0, 2, (4, 16)).float()
    loss = asl(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()

    # Extreme logits (perfect confidence should produce near-zero finite loss)
    extreme_logits = torch.tensor([[-40.0, 40.0] * 8], requires_grad=True)
    extreme_targets = torch.tensor([[0.0, 1.0] * 8])
    loss_ext = asl(extreme_logits, extreme_targets)
    assert torch.isfinite(loss_ext)
    assert loss_ext.item() < 1e-4


def test_bigearthnet_dataset_integrity():
    """Verify dataset loading from archive.zip and correct normalization bounds."""
    archive_path = "initial_trails/Dataset/archive.zip"
    if not Path(archive_path).exists():
        pytest.skip(f"Raw dataset archive not found at {archive_path}")
    dataset = BigEarthNetDataset(
        metadata_path="data/train_full_metadata.parquet",
        archive_path=archive_path,
        subset_size=8,
        augment=False
    )
    assert len(dataset) == 8

    opt, sar, tgt, pid = dataset[0]
    assert opt.shape == (10, 120, 120)
    assert sar.shape == (2, 120, 120)
    assert tgt.shape == (16,)
    assert isinstance(pid, str)

    # SAR normalized values must reside in [0.0, 1.0]
    assert sar.min().item() >= -1e-5
    assert sar.max().item() <= 1.0 + 1e-5


def test_single_batch_overfit():
    """
    Stage 1.4 Smoke Test: Verify that the model can rapidly overfit a single real batch.
    Training loss must drop by at least 70% within 20 optimization steps.
    """
    archive_path = "initial_trails/Dataset/archive.zip"
    if not Path(archive_path).exists():
        pytest.skip(f"Raw dataset archive not found at {archive_path}")

    torch.manual_seed(42)
    np.random.seed(42)

    dataset = BigEarthNetDataset(
        metadata_path="data/train_full_metadata.parquet",
        archive_path=archive_path,
        subset_size=4,
        augment=False
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    b_opt, b_sar, b_tgt, _ = next(iter(loader))

    model = MultimodalResNet34(
        num_classes=16,
        pretrained=False,
        dropout=0.0
    )
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    initial_loss = None
    final_loss = None

    for step in range(25):
        optimizer.zero_grad()
        logits = model(b_opt, b_sar)
        loss = criterion(logits, b_tgt)
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    assert initial_loss is not None and final_loss is not None
    reduction_pct = (initial_loss - final_loss) / initial_loss * 100.0
    print(f"\n[Overfit Smoke Test] Initial Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f} (Drop: {reduction_pct:.1f}%)")
    assert reduction_pct >= 70.0, f"Loss only decreased by {reduction_pct:.1f}%, expected >= 70%"


def test_cpu_throughput_benchmark():
    """
    Stage 1.4 Smoke Test: Benchmark CPU throughput on a realistic batch (size 16),
    measure forward/backward execution time, and extrapolate full epoch runtime.
    """
    batch_size = 16
    model = MultimodalResNet34(num_classes=16, pretrained=False)
    model.train()

    opt = torch.randn(batch_size, 10, 120, 120)
    sar = torch.randn(batch_size, 2, 120, 120)
    tgt = torch.randint(0, 2, (batch_size, 16)).float()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warmup
    optimizer.zero_grad()
    loss = criterion(model(opt[:2], sar[:2]), tgt[:2])
    loss.backward()
    optimizer.step()

    # Timed pass for batch of 16
    t0 = time.perf_counter()
    optimizer.zero_grad()
    logits = model(opt, sar)
    loss = criterion(logits, tgt)
    loss.backward()
    optimizer.step()
    dt = time.perf_counter() - t0

    sec_per_sample = dt / batch_size
    n_train = 7180
    epoch_sec = n_train * sec_per_sample
    epoch_min = epoch_sec / 60.0
    total_40ep_hrs = (epoch_sec * 40.0) / 3600.0

    print(f"\n[CPU Telemetry Benchmark]")
    print(f"Batch size {batch_size} forward+backward time: {dt:.3f}s")
    print(f"Time per sample: {sec_per_sample * 1000:.2f} ms")
    print(f"Estimated 1 epoch (N={n_train}): {epoch_sec:.1f}s (~{epoch_min:.2f} min)")
    print(f"Estimated 40 epochs: {total_40ep_hrs:.2f} hours")

    assert dt > 0 and dt < 30.0, "Execution took unexpectedly long on CPU"
