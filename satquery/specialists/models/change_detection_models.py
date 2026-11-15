"""
Neural Architectures for Bitemporal Change Detection.
Milestone M3.2: Deep Bitemporal Change-Detection Specialist.

Implements:
1. FC-Siam-diff: Fully Convolutional Siamese Difference (Daudt et al., 2018)
2. BIT: Bitemporal Image Transformer (Chen et al., 2021)
"""

import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# =========================================================================
# 1. FC-Siam-diff (Daudt et al., 2018)
# =========================================================================
class ConvBlock(nn.Module):
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


class FCSiamDiff(nn.Module):
    """
    Fully Convolutional Siamese Difference Network for Bitemporal Change Detection.
    Reference: Daudt et al., "Fully Convolutional Siamese Networks for Change Detection", 2018.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super().__init__()
        # Encoder stages (shared between T1 and T2)
        self.enc1 = ConvBlock(in_channels, 16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = ConvBlock(16, 32)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = ConvBlock(32, 64)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.enc4 = ConvBlock(64, 128)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Center bottleneck
        self.bottleneck = ConvBlock(128, 256)

        # Decoder stages (with skip connections from absolute difference)
        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(128 + 128, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(64 + 64, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(32 + 32, 32)

        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(16 + 16, 16)

        # Final prediction head
        self.head = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward_encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        return e1, e2, e3, e4, p4

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        # Extract features from T1 and T2 using shared encoder weights
        e1_1, e2_1, e3_1, e4_1, p4_1 = self.forward_encoder(t1)
        e1_2, e2_2, e3_2, e4_2, p4_2 = self.forward_encoder(t2)

        # Absolute feature differencing at each scale
        diff1 = torch.abs(e1_1 - e1_2)
        diff2 = torch.abs(e2_1 - e2_2)
        diff3 = torch.abs(e3_1 - e3_2)
        diff4 = torch.abs(e4_1 - e4_2)
        diff_p = torch.abs(p4_1 - p4_2)

        # Bottleneck
        b = self.bottleneck(diff_p)

        # Decode with skips
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, diff4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, diff3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, diff2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, diff1], dim=1))

        return self.head(d1)


# =========================================================================
# 2. BIT: Bitemporal Image Transformer (Chen et al., 2021)
# =========================================================================
class SpatialTokenizer(nn.Module):
    """
    Learns spatial attention masks to aggregate continuous feature maps
    into compact semantic tokens (L tokens of dimension C).
    """

    def __init__(self, in_dim: int, num_tokens: int = 64):
        super().__init__()
        self.conv_token = nn.Conv2d(in_dim, num_tokens, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        attn = self.conv_token(x)  # (B, L, H, W)
        attn = attn.view(B, -1, H * W)  # (B, L, N)
        attn = F.softmax(attn, dim=-1)  # Softmax over spatial locations

        feat = x.view(B, C, H * W)  # (B, C, N)
        # Token aggregation: (B, L, N) x (B, N, C) -> (B, L, C)
        tokens = torch.bmm(attn, feat.transpose(1, 2))
        return tokens, attn


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN Self-Attention
        nx = self.norm1(x)
        attn_out, _ = self.attn(nx, nx, nx)
        x = x + attn_out
        # Pre-LN MLP
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # Self-attention
        ntgt = self.norm1(tgt)
        tgt = tgt + self.self_attn(ntgt, ntgt, ntgt)[0]
        # Cross-attention with temporal memory
        ntgt = self.norm2(tgt)
        nmem = self.norm2(memory)
        tgt = tgt + self.cross_attn(ntgt, nmem, nmem)[0]
        # MLP
        tgt = tgt + self.mlp(self.norm3(tgt))
        return tgt


class BitemporalImageTransformer(nn.Module):
    """
    Bitemporal Image Transformer (BIT) for Change Detection.
    Reference: Chen et al., "Remote Sensing Image Change Detection with Transformers", IEEE TGRS 2021.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        num_tokens: int = 64,
        embed_dim: int = 128,
        num_enc_layers: int = 4,
        num_dec_layers: int = 4,
        num_heads: int = 4,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

        # Siamese ResNet-18 backbone (layers 1 to 4)
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained_backbone else None)
        self.backbone_conv1 = resnet.conv1
        self.backbone_bn1 = resnet.bn1
        self.backbone_relu = resnet.relu
        self.backbone_maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 64
        self.layer2 = resnet.layer2  # 128
        self.layer3 = resnet.layer3  # 256
        self.layer4 = resnet.layer4  # 512

        # Projection to embedding dim
        self.proj = nn.Conv2d(512, embed_dim, kernel_size=1)

        # Spatial Tokenizer
        self.tokenizer = SpatialTokenizer(embed_dim, num_tokens=num_tokens)

        # Transformer Encoder (intra-temporal context modeling)
        self.enc_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads=num_heads) for _ in range(num_enc_layers)
        ])

        # Transformer Decoder (cross-temporal difference modeling)
        self.dec_layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads=num_heads) for _ in range(num_dec_layers)
        ])

        # Prediction decoder & upsampling head (16x upsampling)
        # Difference features are fused with restored spatial tokens
        self.head_conv = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),  # 2x
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),  # 4x
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1, bias=False),  # 8x
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1, bias=False),   # 16x
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, num_classes, kernel_size=1),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        # Downsampling factor: 16x
        x = self.backbone_conv1(x)
        x = self.backbone_bn1(x)
        x = self.backbone_relu(x)
        x = self.backbone_maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.proj(x)  # (B, embed_dim, H/16, W/16)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        B, _, H, W = t1.shape

        # 1. Feature extraction
        f1 = self.extract_features(t1)  # (B, C, H', W')
        f2 = self.extract_features(t2)
        _, C, H_feat, W_feat = f1.shape

        # 2. Tokenize into semantic concepts
        tokens1, attn1 = self.tokenizer(f1)  # (B, L, C), (B, L, N)
        tokens2, attn2 = self.tokenizer(f2)  # (B, L, C), (B, L, N)

        # 3. Transformer Encoder (intra-temporal)
        for layer in self.enc_layers:
            tokens1 = layer(tokens1)
            tokens2 = layer(tokens2)

        # 4. Transformer Decoder (cross-temporal difference)
        dec_tokens1 = tokens1
        dec_tokens2 = tokens2
        for layer in self.dec_layers:
            dec_tokens1 = layer(dec_tokens1, tokens2)
            dec_tokens2 = layer(dec_tokens2, tokens1)

        diff_tokens = torch.abs(dec_tokens1 - dec_tokens2)  # (B, L, C)

        # 5. Project tokens back to spatial domain
        # (B, N, L) x (B, L, C) -> (B, N, C)
        combined_attn = 0.5 * (attn1 + attn2)
        spatial_recon = torch.bmm(combined_attn.transpose(1, 2), diff_tokens)
        spatial_recon = spatial_recon.transpose(1, 2).view(B, C, H_feat, W_feat)

        # Difference in CNN feature map
        feat_diff = torch.abs(f1 - f2)

        # Fuse restored tokens with convolutional difference
        fused = torch.cat([spatial_recon, feat_diff], dim=1)  # (B, 2*C, H_feat, W_feat)
        fused = self.head_conv(fused)

        # 6. Upsample to full input resolution
        out = self.upsample(fused)

        # Crop/pad if necessary to match input H, W exactly
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

        return out
