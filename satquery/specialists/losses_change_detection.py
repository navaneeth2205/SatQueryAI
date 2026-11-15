"""
Milestone M3.2 - Step 3: Loss Functions for Multitemporal Change Detection.
Document Code: SATQUERY-SPECIALIST-M3.2-LOSSES

Implements:
1. BCEDiceLoss (Hybrid Binary Cross-Entropy with pos_weight + Soft Dice Loss)
2. BinaryFocalLoss (Lin et al., 2017)
3. FocalTverskyLoss (Abraham & Khan, 2019)
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice Loss for binary segmentation."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(-2, -1))
        denominator = probs.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Hybrid weighted BCE + Soft Dice Loss."""

    def __init__(self, dice_weight: float = 1.0, pos_weight: float = 2.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.pos_weight_val = pos_weight
        self.bce = None
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.bce is None:
            pw = torch.tensor([self.pos_weight_val], device=logits.device)
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return bce_loss + self.dice_weight * dice_loss


class BinaryFocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, eps: float = 1e-7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        # Numerical stability clamping
        probs = torch.clamp(probs, self.eps, 1.0 - self.eps)

        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * torch.pow(1.0 - p_t, self.gamma)

        bce = -torch.log(p_t)
        loss = focal_weight * bce
        return loss.mean()


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for imbalanced medical/remote-sensing segmentation.
    Reference: Abraham & Khan, "A Novel Focal Tversky Loss Function with Improved
    Attention U-Net for Lesion Segmentation", ISBI 2019.
    
    Tversky Index:
    TI = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    Loss = (1 - TI)^gamma
    """

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 1.333,
        smooth: float = 1e-6
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # Flatten spatial dimensions per batch sample
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        tp = (probs_flat * targets_flat).sum(dim=1)
        fp = (probs_flat * (1.0 - targets_flat)).sum(dim=1)
        fn = ((1.0 - probs_flat) * targets_flat).sum(dim=1)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = torch.pow(1.0 - tversky, self.gamma)
        return focal_tversky.mean()
