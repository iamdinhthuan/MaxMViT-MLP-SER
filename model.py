import torch
import torch.nn as nn
import timm
import torchaudio
import librosa
import numpy as np
import cv2
import os

class _ConcatFusion(nn.Module):
    def __init__(self, dim1: int, dim2: int):
        super().__init__()
        self.output_dim = dim1 + dim2

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        return torch.cat((feat1, feat2), dim=1)


class _GatedScalarFusion(nn.Module):
    """
    Learn a scalar gate per-sample, then fuse features.

    Modes:
      - "concat": concat(alpha*f1, (1-alpha)*f2) -> dim1+dim2
      - "sum": alpha*f1 + (1-alpha)*f2 -> dim1 (requires dim1 == dim2)
    """

    def __init__(self, dim1: int, dim2: int, *, mode: str = "concat"):
        super().__init__()
        if mode not in {"concat", "sum"}:
            raise ValueError(f"Invalid mode={mode}. Expected 'concat' or 'sum'.")
        if mode == "sum" and dim1 != dim2:
            raise ValueError(f"mode='sum' requires dim1==dim2, got {dim1} vs {dim2}.")
        self.mode = mode
        self.gate = nn.Linear(dim1 + dim2, 1)
        self.output_dim = (dim1 + dim2) if mode == "concat" else dim1

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        alpha = torch.sigmoid(self.gate(torch.cat((feat1, feat2), dim=1)))  # [B, 1]
        if self.mode == "sum":
            return alpha * feat1 + (1 - alpha) * feat2
        return torch.cat((alpha * feat1, (1 - alpha) * feat2), dim=1)


class _GatedChannelFusion(nn.Module):
    """
    Learn channel-wise gates for each branch.

    If `complementary=True` and dim1==dim2:
      g = sigmoid(W[feat1;feat2])
      fused = concat(g*f1, (1-g)*f2)

    Else:
      g1 = sigmoid(W1[feat1;feat2]), g2 = sigmoid(W2[feat1;feat2])
      fused = concat(g1*f1, g2*f2)
    """

    def __init__(self, dim1: int, dim2: int, *, complementary: bool = True):
        super().__init__()
        self.complementary = complementary and (dim1 == dim2)
        self.gate1 = nn.Linear(dim1 + dim2, dim1)
        self.gate2 = None if self.complementary else nn.Linear(dim1 + dim2, dim2)
        self.output_dim = dim1 + dim2

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        joint = torch.cat((feat1, feat2), dim=1)
        g1 = torch.sigmoid(self.gate1(joint))
        if self.complementary:
            g2 = 1 - g1
        else:
            g2 = torch.sigmoid(self.gate2(joint))
        return torch.cat((g1 * feat1, g2 * feat2), dim=1)


class _TwoTokenAttentionFusion(nn.Module):
    """
    Treat (feat1, feat2) as 2 tokens and run a small TransformerEncoder.
    Output is concatenation of 2 token vectors: [B, 2*d_model].
    """

    def __init__(
        self,
        dim1: int,
        dim2: int,
        *,
        d_model: int = 768,
        n_layers: int = 1,
        n_heads: int = 8,
        dropout: float = 0.1,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.proj1 = nn.Identity() if dim1 == d_model else nn.Linear(dim1, d_model)
        self.proj2 = nn.Identity() if dim2 == d_model else nn.Linear(dim2, d_model)
        self.type_embed = nn.Parameter(torch.zeros(2, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output_dim = 2 * d_model

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        t1 = self.proj1(feat1)
        t2 = self.proj2(feat2)
        tokens = torch.stack((t1, t2), dim=1)  # [B, 2, d_model]
        tokens = tokens + self.type_embed.unsqueeze(0)
        out = self.encoder(tokens)  # [B, 2, d_model]
        return out.reshape(out.shape[0], -1)


class _HadamardFusion(nn.Module):
    """
    A lightweight bilinear-style fusion using elementwise product in a projected space.

    mode:
      - "prod": return z only -> [B, d_proj]
      - "concat": return [p1, p2, z] -> [B, 3*d_proj]
    """

    def __init__(self, dim1: int, dim2: int, *, d_proj: int = 768, mode: str = "concat", dropout: float = 0.0):
        super().__init__()
        if mode not in {"prod", "concat"}:
            raise ValueError(f"Invalid mode={mode}. Expected 'prod' or 'concat'.")
        self.mode = mode
        self.proj1 = nn.Linear(dim1, d_proj)
        self.proj2 = nn.Linear(dim2, d_proj)
        self.drop = nn.Dropout(dropout)
        self.output_dim = d_proj if mode == "prod" else 3 * d_proj

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        p1 = self.drop(self.proj1(feat1))
        p2 = self.drop(self.proj2(feat2))
        z = p1 * p2
        if self.mode == "prod":
            return z
        return torch.cat((p1, p2, z), dim=1)


class _LowRankBilinearFusion(nn.Module):
    """
    Low-rank bilinear interaction:
      z = (U f1) ⊙ (V f2)  in rank-k space, then project to d_out.
    """

    def __init__(self, dim1: int, dim2: int, *, rank: int = 256, d_out: int = 768, dropout: float = 0.0):
        super().__init__()
        self.u = nn.Linear(dim1, rank)
        self.v = nn.Linear(dim2, rank)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(rank, d_out)
        self.output_dim = d_out

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        z = self.drop(self.u(feat1)) * self.drop(self.v(feat2))
        return self.proj(z)


class _MLPHeadBN(nn.Module):
    def __init__(self, in_dim: int, *, hidden_size: int, dropout_rate: float, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _MLPHeadLN(nn.Module):
    def __init__(self, in_dim: int, *, hidden_size: int, dropout_rate: float, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ResidualMLPHeadLN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        *,
        hidden_size: int,
        dropout_rate: float,
        num_classes: int,
        n_blocks: int = 2,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_size)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(hidden_size, hidden_size),
                )
                for _ in range(n_blocks)
            ]
        )
        self.out_norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for block in self.blocks:
            h = h + block(h)
        h = self.out_norm(h)
        return self.classifier(h)


class MaxMViT_MLP(nn.Module):
    def __init__(
        self,
        num_classes=7,
        hidden_size=512,
        dropout_rate=0.2,
        *,
        fusion: str = "concat",
        head: str = "mlp_bn",
        fusion_kwargs: dict | None = None,
        head_kwargs: dict | None = None,
        pretrained: bool = True,
        verbose: bool = True,
    ):
        """
        MaxMViT and MViTv2 Fusion Network with Multilayer Perceptron (MaxMViT-MLP).
        
        Args:
            num_classes (int): Number of emotion classes (e.g., 7 for Emo-DB).
            hidden_size (int): Number of hidden nodes in MLP (default 512).
            dropout_rate (float): Dropout rate (default 0.2).
        """
        super(MaxMViT_MLP, self).__init__()
        
        # --- Path 1: CQT + MaxViT ---
        self.maxvit = timm.create_model('maxvit_base_tf_224', pretrained=pretrained, num_classes=0)
        # --- Path 2: MEL + MViTv2 ---
        self.mvitv2 = timm.create_model('mvitv2_base', pretrained=pretrained, num_classes=0)

        if verbose:
            print(f"Initialized MaxViT: {self.maxvit.default_cfg['architecture']}")
        
        # Calculate feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224) 
            maxvit_dim = self.maxvit(dummy_input).shape[1]
            mvitv2_dim = self.mvitv2(dummy_input).shape[1]
            
        fusion_kwargs = fusion_kwargs or {}
        head_kwargs = head_kwargs or {}

        fusion = fusion.lower().strip()
        if fusion == "concat":
            self.fusion = _ConcatFusion(maxvit_dim, mvitv2_dim)
        elif fusion in {"gated_scalar", "gate_scalar"}:
            self.fusion = _GatedScalarFusion(maxvit_dim, mvitv2_dim, **fusion_kwargs)
        elif fusion in {"gated_channel", "gate_channel"}:
            self.fusion = _GatedChannelFusion(maxvit_dim, mvitv2_dim, **fusion_kwargs)
        elif fusion in {"attn2token", "two_token_attn", "two_token_attention"}:
            self.fusion = _TwoTokenAttentionFusion(maxvit_dim, mvitv2_dim, **fusion_kwargs)
        elif fusion in {"hadamard", "bilinear_hadamard"}:
            self.fusion = _HadamardFusion(maxvit_dim, mvitv2_dim, **fusion_kwargs)
        elif fusion in {"lowrank_bilinear", "bilinear_lowrank"}:
            self.fusion = _LowRankBilinearFusion(maxvit_dim, mvitv2_dim, **fusion_kwargs)
        else:
            raise ValueError(
                f"Unknown fusion='{fusion}'. Supported: concat, gated_scalar, gated_channel, attn2token, hadamard, lowrank_bilinear"
            )

        fusion_dim = int(self.fusion.output_dim)

        head = head.lower().strip()
        if head in {"mlp_bn", "bn"}:
            self.mlp = _MLPHeadBN(fusion_dim, hidden_size=hidden_size, dropout_rate=dropout_rate, num_classes=num_classes)
        elif head in {"mlp_ln", "ln"}:
            self.mlp = _MLPHeadLN(fusion_dim, hidden_size=hidden_size, dropout_rate=dropout_rate, num_classes=num_classes)
        elif head in {"mlp_residual_ln", "residual_ln", "resmlp_ln"}:
            self.mlp = _ResidualMLPHeadLN(
                fusion_dim,
                hidden_size=hidden_size,
                dropout_rate=dropout_rate,
                num_classes=num_classes,
                **head_kwargs,
            )
        else:
            raise ValueError(f"Unknown head='{head}'. Supported: mlp_bn, mlp_ln, mlp_residual_ln")

    def forward(self, cqt, mel):
        """
        Forward pass.
        
        Args:
           cqt (torch.Tensor): CQT Spectrogram [Batch, 3, H, W]
           mel (torch.Tensor): Mel-STFT Spectrogram [Batch, 3, H, W]
        """
        # Expand 1 channel to 3 channels for backbone compatibility
        if cqt.size(1) == 1:
            cqt = cqt.repeat(1, 3, 1, 1)
        if mel.size(1) == 1:
            mel = mel.repeat(1, 3, 1, 1)
        
        # Resize to 224x224 (MaxViT requires input divisible by 32)
        if cqt.shape[-1] != 224:
             cqt = torch.nn.functional.interpolate(cqt, size=(224, 224), mode='bilinear', align_corners=False)
        if mel.shape[-1] != 224:
             mel = torch.nn.functional.interpolate(mel, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Path 1: CQT -> MaxViT
        feat_maxvit = self.maxvit(cqt)  # [B, 768]
        
        # Path 2: MEL -> MViTv2
        feat_mvitv2 = self.mvitv2(mel)  # [B, 768]
        
        fused = self.fusion(feat_maxvit, feat_mvitv2)
        
        # MLP Classification
        logits = self.mlp(fused)
        
        return logits


def get_optimizer(model, lr=0.02):
    """
    Returns the optimizers as specified in the paper:
    - MaxViT + MLP: Adam
    - MViTv2: RAdam
    """
    maxvit_params = list(model.maxvit.parameters())
    mvitv2_params = list(model.mvitv2.parameters())
    mlp_params = list(model.mlp.parameters())
    fusion_params = list(model.fusion.parameters()) if hasattr(model, "fusion") else []
    
    # Optimizer 1: MaxViT + MLP -> Adam
    optimizer1 = torch.optim.Adam(maxvit_params + mlp_params + fusion_params, lr=lr)
    
    # Optimizer 2: MViTv2 -> RAdam
    optimizer2 = torch.optim.RAdam(mvitv2_params, lr=lr)
    
    return [optimizer1, optimizer2]
