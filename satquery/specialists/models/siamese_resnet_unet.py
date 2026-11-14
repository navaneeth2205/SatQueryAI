"""
Milestone M3.2 - Step 3: Siamese ResNet-18 U-Net Specialist Architecture.
Document Code: SATQUERY-SPECIALIST-M3.2-SIAM-RESNET18-UNET

Implements:
1. Twin Siamese ResNet-18 Encoders (scratch or ImageNet pretrained)
2. Multiscale bitemporal fusion:
   - Absolute feature difference (|f1 - f2|)
   - Richer temporal fusion (Concat + Difference: [|f1 - f2|, f1, f2] with 1x1 projection)
3. Spatial-Channel Squeeze & Excitation (scSE) attention blocks (Roy et al., 2018)
4. Full-resolution U-Net decoder with multi-scale skip connections
"""

import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConvBlock(nn.Module):
    """Standard double 3x3 convolution block with BatchNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SCSEBlock(nn.Module):
    """
    Spatial and Channel Squeeze & Excitation (scSE) Block.
    Reference: Roy et al., "Concurrent Spatial and Channel 'Squeeze & Excitation'
    in Fully Convolutional Networks", MICCAI 2018 / IEEE TMI 2018.
    """

    def __init__(self, in_ch: int, r: int = 16):
        super().__init__()
        # Channel Squeeze and Spatial Excitation (sSE)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_ch, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # Spatial Squeeze and Channel Excitation (cSE)
        reduced_ch = max(in_ch // r, 8)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, reduced_ch, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_ch, in_ch, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Concurrent spatial and channel excitation
        x_s = x * self.spatial_gate(x)
        x_c = x * self.channel_gate(x)
        return x_s + x_c


class TemporalFusionModule(nn.Module):
    """
    Multiscale Temporal Fusion Module.
    Supports:
    - 'diff': Absolute feature difference |f1 - f2|
    - 'diff_cat': Concatenation of difference and raw features [|f1 - f2|, f1, f2]
                  projected back to channel dimension C via 1x1 conv + BN + ReLU.
    """

    def __init__(self, in_ch: int, mode: str = "diff"):
        super().__init__()
        self.mode = mode.lower()
        if self.mode == "diff_cat":
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch * 3, in_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True)
            )
        elif self.mode == "diff":
            self.proj = nn.Identity()
        else:
            raise ValueError(f"Unknown fusion mode: {mode}. Expected 'diff' or 'diff_cat'.")

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(f1 - f2)
        if self.mode == "diff_cat":
            cat = torch.cat([diff, f1, f2], dim=1)
            return self.proj(cat)
        return diff


class SiameseResNet18UNet(nn.Module):
    """
    Siamese ResNet-18 U-Net for Bitemporal Change Detection.

    Args:
        in_channels: Input image channels (3 for RGB).
        num_classes: Output classes (1 for binary change segmentation).
        pretrained: If True, initialize twin encoders from ImageNet pretrained weights.
        fusion_mode: 'diff' (|f1 - f2|) or 'diff_cat' ([|f1 - f2|, f1, f2] with 1x1 conv).
        use_scse: If True, append scSE attention blocks after each decoder stage.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        pretrained: bool = True,
        fusion_mode: str = "diff",
        use_scse: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.fusion_mode = fusion_mode.lower()
        self.use_scse = use_scse

        # 1. Twin Shared Encoder (ResNet-18)
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Handle arbitrary input channels if not 3
        if in_channels != 3:
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(
                in_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            if pretrained and in_channels > 3:
                # Copy RGB weights and initialize extra channels with average
                with torch.no_grad():
                    new_conv.weight[:, :3] = old_conv.weight
                    for c in range(3, in_channels):
                        new_conv.weight[:, c:c+1] = old_conv.weight.mean(dim=1, keepdim=True)
            backbone.conv1 = new_conv

        # Encoder stages
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu
        )  # (B, 64, H/2, W/2)
        self.maxpool = backbone.maxpool  # (B, 64, H/4, W/4)
        self.layer1 = backbone.layer1    # (B, 64, H/4, W/4)
        self.layer2 = backbone.layer2    # (B, 128, H/8, W/8)
        self.layer3 = backbone.layer3    # (B, 256, H/16, W/16)
        self.layer4 = backbone.layer4    # (B, 512, H/32, W/32)

        # 2. Multiscale Fusion Modules
        self.fuse0 = TemporalFusionModule(64, mode=fusion_mode)
        self.fuse1 = TemporalFusionModule(64, mode=fusion_mode)
        self.fuse2 = TemporalFusionModule(128, mode=fusion_mode)
        self.fuse3 = TemporalFusionModule(256, mode=fusion_mode)
        self.fuse4 = TemporalFusionModule(512, mode=fusion_mode)

        # 3. Center Bottleneck
        self.bottleneck = ConvBlock(512, 512)

        # 4. Decoder with multiscale skips
        # Stage 4: H/32 -> H/16
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(256 + 256, 256)
        self.scse4 = SCSEBlock(256) if use_scse else nn.Identity()

        # Stage 3: H/16 -> H/8
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(128 + 128, 128)
        self.scse3 = SCSEBlock(128) if use_scse else nn.Identity()

        # Stage 2: H/8 -> H/4
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(64 + 64, 64)
        self.scse2 = SCSEBlock(64) if use_scse else nn.Identity()

        # Stage 1: H/4 -> H/2
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64 + 64, 64)
        self.scse1 = SCSEBlock(64) if use_scse else nn.Identity()

        # Final Stage: H/2 -> H/1
        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec0 = ConvBlock(32, 32)
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward_encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        e0 = self.stem(x)       # (B, 64, H/2, W/2)
        p0 = self.maxpool(e0)   # (B, 64, H/4, W/4)
        e1 = self.layer1(p0)    # (B, 64, H/4, W/4)
        e2 = self.layer2(e1)    # (B, 128, H/8, W/8)
        e3 = self.layer3(e2)    # (B, 256, H/16, W/16)
        e4 = self.layer4(e3)    # (B, 512, H/32, W/32)
        return e0, e1, e2, e3, e4

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        # Extract features from T1 and T2 using shared twin encoder
        t1_e0, t1_e1, t1_e2, t1_e3, t1_e4 = self.forward_encoder(t1)
        t2_e0, t2_e1, t2_e2, t2_e3, t2_e4 = self.forward_encoder(t2)

        # Temporal fusion at each scale
        f0 = self.fuse0(t1_e0, t2_e0)  # (B, 64, H/2, W/2)
        f1 = self.fuse1(t1_e1, t2_e1)  # (B, 64, H/4, W/4)
        f2 = self.fuse2(t1_e2, t2_e2)  # (B, 128, H/8, W/8)
        f3 = self.fuse3(t1_e3, t2_e3)  # (B, 256, H/16, W/16)
        f4 = self.fuse4(t1_e4, t2_e4)  # (B, 512, H/32, W/32)

        # Bottleneck
        b = self.bottleneck(f4)

        # Decoder stages with skips and optional scSE attention
        d4 = self.up4(b)
        d4 = torch.cat([d4, f3], dim=1)
        d4 = self.dec4(d4)
        d4 = self.scse4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, f2], dim=1)
        d3 = self.dec3(d3)
        d3 = self.scse3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, f1], dim=1)
        d2 = self.dec2(d2)
        d2 = self.scse2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, f0], dim=1)
        d1 = self.dec1(d1)
        d1 = self.scse1(d1)

        d0 = self.up0(d1)
        d0 = self.dec0(d0)
        logits = self.head(d0)

        return logits
