"""
Multimodal Dual-Branch ResNet-34 Architecture for Land Cover Classification.

Implements the Milestone M3.0 / M3.1 specification (Deliverable F.1):
- Optical Branch: ResNet-34 backbone with 10 Sentinel-2 multispectral bands.
  Pretrained ImageNet weight transfer:
    * Channel corresponding to B04 (Red, ~665 nm) <- ImageNet Red (conv1[:, 0:1])
    * Channel corresponding to B03 (Green, ~560 nm) <- ImageNet Green (conv1[:, 1:2])
    * Channel corresponding to B02 (Blue, ~490 nm) <- ImageNet Blue (conv1[:, 2:3])
    * Remaining 7 channels (B05-B08, B8A, B11, B12) <- He normal initialization (gain=sqrt(2))
- SAR Branch: ResNet-34 backbone with 2 Sentinel-1 bands [VH, VV].
  Initialized from scratch with He normal initialization.
- Cross-Attention Fusion:
  Optical and SAR spatial feature maps (512 channels) coupled via 8-head
  multi-head cross-attention (d_k = 64, total dim = 512), followed by
  LayerNorm, 2-layer MLP residual block, and global average pooling.
- 16-class sigmoid classification head outputting raw logits.
"""

from typing import Dict, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


class CrossAttentionFusionBlock(nn.Module):
    """
    Bidirectional multi-head cross-attention fusion block coupling
    Optical and SAR spatial representations.
    
    Channels: 512, Heads: 8, Key/Query/Value dimension: 64 (8 * 64 = 512).
    Followed by LayerNorm and a 2-layer MLP residual block with GELU.
    """
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Optical queries attend to SAR keys/values
        self.cross_attn_opt_to_sar = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        # SAR queries attend to Optical keys/values
        self.cross_attn_sar_to_opt = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        
        # 2-layer MLP residual block
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        feat_opt: torch.Tensor,
        feat_sar: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            feat_opt: Optical spatial sequence tensor (B, S, C)
            feat_sar: SAR spatial sequence tensor (B, S, C)
        Returns:
            Fused spatial sequence tensor (B, S, C)
        """
        # Cross attention: optical query -> SAR key/val
        attn_opt, _ = self.cross_attn_opt_to_sar(
            query=feat_opt,
            key=feat_sar,
            value=feat_sar
        )
        # Cross attention: SAR query -> optical key/val
        attn_sar, _ = self.cross_attn_sar_to_opt(
            query=feat_sar,
            key=feat_opt,
            value=feat_opt
        )

        # Residual accumulation from both pathways
        fused = (feat_opt + attn_opt) + (feat_sar + attn_sar)
        fused = self.norm1(fused)

        # MLP residual block
        out = fused + self.mlp(fused)
        out = self.norm2(out)
        return out


class MultimodalResNet34(nn.Module):
    """
    Dual-branch ResNet-34 with Cross-Attention Fusion for Multimodal Land Cover.
    """

    def __init__(
        self,
        num_classes: int = 16,
        optical_channels: int = 10,
        sar_channels: int = 2,
        pretrained: bool = True,
        channel_order: str = "b02_first",
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            num_classes: Number of target land cover classes (16 for BigEarthNet).
            optical_channels: Number of Sentinel-2 channels (10).
            sar_channels: Number of Sentinel-1 SAR channels (2: VH, VV).
            pretrained: Whether to initialize optical backbone from ImageNet-1k weights.
            channel_order: 'b02_first' ([B02, B03, B04, ...]) or 'rgb_first' ([B04, B03, B02, ...]).
            embed_dim: Feature embedding dimension (512).
            num_heads: Attention heads (8).
            dropout: Dropout probability (0.1).
        """
        super().__init__()
        self.num_classes = num_classes
        self.optical_channels = optical_channels
        self.sar_channels = sar_channels
        self.channel_order = channel_order
        self.embed_dim = embed_dim

        # 1. Optical Branch Backbone (ResNet-34)
        if pretrained:
            weights = tv_models.ResNet34_Weights.DEFAULT
            base_opt = tv_models.resnet34(weights=weights)
        else:
            base_opt = tv_models.resnet34(weights=None)

        # Modified conv1 for 10 optical channels
        orig_conv1_weight = base_opt.conv1.weight.data.clone()  # Shape: (64, 3, 7, 7)
        self.opt_conv1 = nn.Conv2d(
            in_channels=optical_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self._init_optical_conv1(orig_conv1_weight, pretrained)

        self.opt_bn1 = base_opt.bn1
        self.opt_relu = base_opt.relu
        self.opt_maxpool = base_opt.maxpool
        self.opt_layer1 = base_opt.layer1  # 64 ch
        self.opt_layer2 = base_opt.layer2  # 128 ch
        self.opt_layer3 = base_opt.layer3  # 256 ch
        self.opt_layer4 = base_opt.layer4  # 512 ch

        # 2. SAR Branch Backbone (ResNet-34 initialized from scratch)
        base_sar = tv_models.resnet34(weights=None)
        self.sar_conv1 = nn.Conv2d(
            in_channels=sar_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        # He normal initialization for SAR conv1
        nn.init.kaiming_normal_(self.sar_conv1.weight, mode='fan_out', nonlinearity='relu')

        self.sar_bn1 = base_sar.bn1
        self.sar_relu = base_sar.relu
        self.sar_maxpool = base_sar.maxpool
        self.sar_layer1 = base_sar.layer1  # 64 ch
        self.sar_layer2 = base_sar.layer2  # 128 ch
        self.sar_layer3 = base_sar.layer3  # 256 ch
        self.sar_layer4 = base_sar.layer4  # 512 ch

        # 3. Cross-Attention Multimodal Fusion
        self.fusion = CrossAttentionFusionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=2.0,
            dropout=dropout
        )

        # 4. Classification Head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

        # Record mapping audit for transparency and tests
        self._mapping_audit = self._build_channel_mapping_audit()

    def _init_optical_conv1(self, orig_weight: torch.Tensor, pretrained: bool) -> None:
        """
        Map pretrained RGB weights and initialize multispectral channels.
        
        Protocol Specification (Deliverable F.1):
        - Sentinel-2 B04 (Red, ~665 nm) <- ImageNet Red (orig_weight[:, 0:1])
        - Sentinel-2 B03 (Green, ~560 nm) <- ImageNet Green (orig_weight[:, 1:2])
        - Sentinel-2 B02 (Blue, ~490 nm) <- ImageNet Blue (orig_weight[:, 2:3])
        - Remaining 7 multispectral channels <- He normal (gain=sqrt(2))
        """
        # First initialize all 10 channels with He normal
        nn.init.kaiming_normal_(self.opt_conv1.weight, mode='fan_out', nonlinearity='relu')

        if not pretrained:
            return

        with torch.no_grad():
            if self.channel_order == "b02_first":
                # S2: 0: B02 (Blue), 1: B03 (Green), 2: B04 (Red), 3-9: B05..B12
                self.opt_conv1.weight.data[:, 0:1, :, :] = orig_weight[:, 2:3, :, :]  # Blue
                self.opt_conv1.weight.data[:, 1:2, :, :] = orig_weight[:, 1:2, :, :]  # Green
                self.opt_conv1.weight.data[:, 2:3, :, :] = orig_weight[:, 0:1, :, :]  # Red
            elif self.channel_order == "rgb_first":
                # S2: 0: B04 (Red), 1: B03 (Green), 2: B02 (Blue), 3-9: B05..B12
                self.opt_conv1.weight.data[:, 0:1, :, :] = orig_weight[:, 0:1, :, :]  # Red
                self.opt_conv1.weight.data[:, 1:2, :, :] = orig_weight[:, 1:2, :, :]  # Green
                self.opt_conv1.weight.data[:, 2:3, :, :] = orig_weight[:, 2:3, :, :]  # Blue
            else:
                raise ValueError(f"Unknown channel_order: {self.channel_order}")

    def _build_channel_mapping_audit(self) -> Dict[str, str]:
        if self.channel_order == "b02_first":
            return {
                "channel_0": "S2 B02 (Blue) <- ImageNet Blue filter (conv1[:, 2:3])",
                "channel_1": "S2 B03 (Green) <- ImageNet Green filter (conv1[:, 1:2])",
                "channel_2": "S2 B04 (Red) <- ImageNet Red filter (conv1[:, 0:1])",
                "channels_3_9": "S2 B05..B12 <- He normal initialization (gain=sqrt(2))",
                "sar_channels": "S1 VH, VV <- He normal initialization (gain=sqrt(2))"
            }
        else:
            return {
                "channel_0": "S2 B04 (Red) <- ImageNet Red filter (conv1[:, 0:1])",
                "channel_1": "S2 B03 (Green) <- ImageNet Green filter (conv1[:, 1:2])",
                "channel_2": "S2 B02 (Blue) <- ImageNet Blue filter (conv1[:, 2:3])",
                "channels_3_9": "S2 B05..B12 <- He normal initialization (gain=sqrt(2))",
                "sar_channels": "S1 VH, VV <- He normal initialization (gain=sqrt(2))"
            }

    @property
    def mapping_audit(self) -> Dict[str, str]:
        return self._mapping_audit

    def forward_features(
        self,
        x_opt: torch.Tensor,
        x_sar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract spatial feature maps from Optical and SAR backbones.
        
        Args:
            x_opt: Optical tensor (B, 10, H, W)
            x_sar: SAR tensor (B, 2, H, W)
        Returns:
            Tuple of (feat_opt, feat_sar), each of shape (B, 512, H_feat, W_feat)
        """
        # Optical pathway
        opt = self.opt_conv1(x_opt)
        opt = self.opt_bn1(opt)
        opt = self.opt_relu(opt)
        opt = self.opt_maxpool(opt)
        opt = self.opt_layer1(opt)
        opt = self.opt_layer2(opt)
        opt = self.opt_layer3(opt)
        opt = self.opt_layer4(opt)  # (B, 512, H', W')

        # SAR pathway
        sar = self.sar_conv1(x_sar)
        sar = self.sar_bn1(sar)
        sar = self.sar_relu(sar)
        sar = self.sar_maxpool(sar)
        sar = self.sar_layer1(sar)
        sar = self.sar_layer2(sar)
        sar = self.sar_layer3(sar)
        sar = self.sar_layer4(sar)  # (B, 512, H', W')

        return opt, sar

    def forward_with_embeddings(
        self,
        x_opt: torch.Tensor,
        x_sar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both classification logits and 512-dim fused embeddings.
        
        Returns:
            logits: (B, num_classes)
            embeddings: (B, embed_dim)
        """
        # 1. Feature extraction
        opt_spatial, sar_spatial = self.forward_features(x_opt, x_sar)
        B, C, H, W = opt_spatial.shape

        # 2. Reshape to spatial sequences: (B, H*W, C)
        opt_seq = opt_spatial.flatten(2).transpose(1, 2)  # (B, S, 512)
        sar_seq = sar_spatial.flatten(2).transpose(1, 2)  # (B, S, 512)

        # 3. Cross-attention multimodal fusion
        fused_seq = self.fusion(opt_seq, sar_seq)  # (B, S, 512)

        # 4. Global average pooling over spatial tokens
        embeddings = fused_seq.mean(dim=1)  # (B, 512)

        # 5. Logits
        logits = self.head(embeddings)  # (B, 16)
        return logits, embeddings

    def forward(
        self,
        x_opt: torch.Tensor,
        x_sar: torch.Tensor
    ) -> torch.Tensor:
        """
        Standard forward pass returning classification logits.
        """
        logits, _ = self.forward_with_embeddings(x_opt, x_sar)
        return logits
