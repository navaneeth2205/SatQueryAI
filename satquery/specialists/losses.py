"""
Loss formulations for Deep Multimodal Land Cover Classification.

Implements the Milestone M3.0 / M3.1 specification (Deliverable F.1):
1. Normalized Class-Balanced BCE (CB-BCE):
   - Effective-number class weights (Cui et al., 2019) with beta = 0.999.
   - Evaluated strictly over the static training split (N = 7,180).
   - Zero-support fallback rule (w_raw = 1.0) preventing NaNs.
   - Fixed normalized positive weights registered as a non-trainable constant buffer.
   - Mean reduction over 16B elements matching PyTorch BCEWithLogitsLoss.
2. Multi-label Asymmetric Loss (ASL):
   - Multi-label focal loss with asymmetric focusing parameters (gamma+ = 0, gamma- = 4, m = 0.05).
   - Mean reduction over 16B elements ensuring identical gradient scale parity with CB-BCE.
"""

from typing import Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_balanced_weights(
    pos_counts: Union[np.ndarray, torch.Tensor, list],
    total_samples: int = 7180,
    beta: float = 0.999,
    eps: float = 1e-5
) -> torch.Tensor:
    """
    Compute static normalized class-balanced positive weights for multi-label classification.
    
    Equations (Deliverable F.1, lines 1087-1102):
        E_{n, c, +} = (1 - beta^{N_{c, +}}) / (1 - beta)
        w_{c, +}^{raw} = 1 / E_{n, c, +}  (or 1.0 if N_{c, +} == 0)
        w_{c, +}^{norm} = ((w_{c, +}^{raw} + eps) / sum(w_{j, +}^{raw} + eps)) * num_classes
    """
    if isinstance(pos_counts, torch.Tensor):
        counts = pos_counts.detach().cpu().numpy()
    else:
        counts = np.array(pos_counts, dtype=np.float64)

    num_classes = len(counts)
    raw_weights = np.zeros(num_classes, dtype=np.float64)

    for c in range(num_classes):
        n_pos = counts[c]
        if n_pos <= 0:
            # Zero-support fallback rule: finite assignment prevents 0/0 division and NaNs
            raw_weights[c] = 1.0
        else:
            eff_num = (1.0 - np.power(beta, n_pos)) / (1.0 - beta)
            raw_weights[c] = 1.0 / max(eff_num, 1e-8)

    # Normalize weights so their sum equals num_classes (mean weight is 1.0)
    w_sum = np.sum(raw_weights + eps)
    norm_weights = ((raw_weights + eps) / w_sum) * num_classes

    return torch.tensor(norm_weights, dtype=torch.float32)


class ClassBalancedBCEWithLogitsLoss(nn.Module):
    """
    Normalized Class-Balanced Binary Cross-Entropy with Logits Loss.
    
    Averages over both batch dimension B and class dimension C=16:
    L_{CB-BCE} = (1 / 16B) sum_{i=1}^B sum_{c=1}^{16} l_{i, c}
    """

    def __init__(
        self,
        pos_counts: Union[np.ndarray, torch.Tensor, list],
        total_samples: int = 7180,
        beta: float = 0.999,
        eps: float = 1e-5
    ):
        super().__init__()
        weights = compute_class_balanced_weights(
            pos_counts=pos_counts,
            total_samples=total_samples,
            beta=beta,
            eps=eps
        )
        self.register_buffer("pos_weight", weights)
        self.num_classes = len(weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predicted class logits (B, num_classes)
            targets: Binary target multi-hot labels (B, num_classes), float in {0, 1}
        Returns:
            Scalar loss with mean reduction (1 / (B * C))
        """
        targets = targets.to(dtype=logits.dtype, device=logits.device)
        return F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="mean"
        )


class AsymmetricLossWithLogits(nn.Module):
    """
    Multi-Label Asymmetric Loss (ASL) for Positive-Negative Imbalance.
    
    Formula (Deliverable F.1, lines 1103-1105):
    L_{ASL} = - (1 / 16B) sum_{i=1}^B sum_{c=1}^{16} [
        y_{i, c} * (1 - p_{i, c})^{gamma_+} * log(p_{i, c}) +
        (1 - y_{i, c}) * (p_{i, c, m})^{gamma_-} * log(1 - p_{i, c, m})
    ]
    with gamma_+ = 0, gamma_- = 4, m = 0.05.
    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip_margin: float = 0.05,
        eps: float = 1e-7
    ):
        super().__init__()
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.clip_margin = float(clip_margin)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predicted class logits (B, num_classes)
            targets: Binary target multi-hot labels (B, num_classes), float in {0, 1}
        Returns:
            Scalar loss with mean reduction (1 / (B * C))
        """
        targets = targets.to(dtype=logits.dtype, device=logits.device)

        # Probabilities via sigmoid
        probs = torch.sigmoid(logits)

        # 1. Positive loss component
        # When gamma_pos == 0: (1 - p)^0 = 1, loss is simply -log(p)
        probs_pos = torch.clamp(probs, min=self.eps, max=1.0)
        pos_weight = torch.pow(1.0 - probs_pos, self.gamma_pos) if self.gamma_pos > 0 else 1.0
        loss_pos = -targets * pos_weight * torch.log(probs_pos)

        # 2. Negative loss component with asymmetric margin shifting
        # p_m = max(p - m, 0)
        probs_neg_shifted = torch.clamp(probs - self.clip_margin, min=0.0, max=1.0)
        probs_neg_sub = torch.clamp(1.0 - probs_neg_shifted, min=self.eps, max=1.0)
        neg_weight = torch.pow(probs_neg_shifted, self.gamma_neg) if self.gamma_neg > 0 else 1.0
        loss_neg = -(1.0 - targets) * neg_weight * torch.log(probs_neg_sub)

        # Mean reduction over all B * C elements (identical 1/(16B) scale to CB-BCE)
        loss = loss_pos + loss_neg
        return torch.mean(loss)
