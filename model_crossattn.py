"""
MaxMViT-MLP with PROPER Cross-Attention Fusion

FIXED VERSION: Uses spatial features (before global pooling) for meaningful cross-attention.

Key Fix:
    - Extract intermediate spatial features from backbones [B, C, H, W]
    - Reshape to sequence [B, H*W, C] for cross-attention
    - Cross-attention between spatial tokens from both modalities
    - Then pool and classify

Architecture:
    CQT → MaxViT (spatial) → [B, H*W, dim] ─┐
                                             ├→ Cross-Attention → Pool → MLP → Class
    Mel → MViTv2 (spatial) → [B, H*W, dim] ─┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-Head Cross-Attention for sequence-to-sequence attention.
    
    Query from modality A attends to Keys/Values from modality B.
    """
    
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, query, key_value):
        """
        Args:
            query: [B, N_q, dim] - Query sequence from modality A
            key_value: [B, N_kv, dim] - Key/Value sequence from modality B
            
        Returns:
            out: [B, N_q, dim] - Attended features
            attn_weights: [B, num_heads, N_q, N_kv] - Attention map
        """
        B, N_q, C = query.shape
        N_kv = key_value.shape[1]
        
        # Project
        Q = self.q_proj(query)      # [B, N_q, dim]
        K = self.k_proj(key_value)  # [B, N_kv, dim]
        V = self.v_proj(key_value)  # [B, N_kv, dim]
        
        # Reshape for multi-head: [B, N, dim] -> [B, num_heads, N, head_dim]
        Q = Q.view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention: [B, heads, N_q, head_dim] @ [B, heads, head_dim, N_kv]
        attn = (Q @ K.transpose(-2, -1)) * self.scale  # [B, heads, N_q, N_kv]
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.attn_drop(attn_weights)
        
        # Apply attention: [B, heads, N_q, N_kv] @ [B, heads, N_kv, head_dim]
        out = attn_weights @ V  # [B, heads, N_q, head_dim]
        
        # Reshape back: [B, heads, N_q, head_dim] -> [B, N_q, dim]
        out = out.transpose(1, 2).contiguous().view(B, N_q, -1)
        
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out, attn_weights


class CrossAttentionBlock(nn.Module):
    """
    Transformer-style Cross-Attention Block with FFN.
    
    Structure:
        x = x + CrossAttn(norm(x), norm(context))
        x = x + FFN(norm(x))
    """
    
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0.):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(dim)
        
        self.cross_attn = MultiHeadCrossAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop)
        )
        
    def forward(self, x, context):
        """
        Args:
            x: [B, N, dim] - Query sequence
            context: [B, M, dim] - Context sequence to attend to
            
        Returns:
            x: [B, N, dim] - Updated features
            attn: Attention weights for visualization
        """
        # Cross-attention with residual
        x_norm = self.norm1(x)
        context_norm = self.norm_context(context)
        attended, attn = self.cross_attn(x_norm, context_norm)
        x = x + attended
        
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        
        return x, attn


class BidirectionalCrossAttention(nn.Module):
    """
    Bidirectional Cross-Attention between two modalities.
    
    Both modalities attend to each other:
        - CQT tokens attend to Mel tokens
        - Mel tokens attend to CQT tokens
    """
    
    def __init__(self, dim, num_heads=8, num_layers=2, mlp_ratio=4., 
                 drop=0., attn_drop=0.):
        super().__init__()
        
        self.num_layers = num_layers
        
        # Cross-attention layers for each direction
        self.cqt_to_mel_layers = nn.ModuleList([
            CrossAttentionBlock(dim, num_heads, mlp_ratio, drop=drop, attn_drop=attn_drop)
            for _ in range(num_layers)
        ])
        
        self.mel_to_cqt_layers = nn.ModuleList([
            CrossAttentionBlock(dim, num_heads, mlp_ratio, drop=drop, attn_drop=attn_drop)
            for _ in range(num_layers)
        ])
        
    def forward(self, feat_cqt, feat_mel):
        """
        Args:
            feat_cqt: [B, N, dim] - CQT spatial features
            feat_mel: [B, M, dim] - Mel spatial features
            
        Returns:
            feat_cqt: Updated CQT features
            feat_mel: Updated Mel features
            attn_info: Attention weights for analysis
        """
        attn_cqt_to_mel = []
        attn_mel_to_cqt = []
        
        for i in range(self.num_layers):
            # CQT attends to Mel
            feat_cqt, attn1 = self.cqt_to_mel_layers[i](feat_cqt, feat_mel)
            attn_cqt_to_mel.append(attn1)
            
            # Mel attends to CQT
            feat_mel, attn2 = self.mel_to_cqt_layers[i](feat_mel, feat_cqt)
            attn_mel_to_cqt.append(attn2)
        
        attn_info = {
            'cqt_to_mel': attn_cqt_to_mel,
            'mel_to_cqt': attn_mel_to_cqt
        }
        
        return feat_cqt, feat_mel, attn_info


class SpatialFeatureExtractor(nn.Module):
    """
    Wrapper to extract spatial features from timm models BEFORE global pooling.
    """
    
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        
        # Create model without classification head
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # Get feature info
        self.model_name = model_name
        
    def forward(self, x):
        """
        Extract spatial features before global pooling.
        
        Returns:
            spatial_features: [B, H*W, C] - Sequence of spatial tokens
            pooled_features: [B, C] - Global pooled features
        """
        # For MaxViT and MViTv2, we need to get features before the final pool
        # Use forward_features which returns before pooling
        
        if 'maxvit' in self.model_name:
            # MaxViT: forward_features returns [B, C, H, W]
            x = self.model.forward_features(x)
            B, C, H, W = x.shape
            spatial = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
            pooled = x.mean(dim=[2, 3])  # [B, C]
            
        elif 'mvitv2' in self.model_name:
            # MViTv2: forward_features returns [B, N, C] already
            x = self.model.forward_features(x)
            if x.dim() == 3:
                spatial = x  # Already [B, N, C]
                pooled = x.mean(dim=1)  # [B, C]
            else:
                # If [B, C, H, W]
                B, C, H, W = x.shape
                spatial = x.flatten(2).transpose(1, 2)
                pooled = x.mean(dim=[2, 3])
        else:
            # Generic fallback
            x = self.model.forward_features(x)
            if x.dim() == 4:
                B, C, H, W = x.shape
                spatial = x.flatten(2).transpose(1, 2)
                pooled = x.mean(dim=[2, 3])
            else:
                spatial = x
                pooled = x.mean(dim=1)
                
        return spatial, pooled


class MaxMViT_MLP_CrossAttn(nn.Module):
    """
    MaxMViT-MLP with PROPER Cross-Attention Fusion.
    
    Uses spatial features (before pooling) for meaningful cross-attention.
    
    Architecture:
        CQT → MaxViT → spatial [B, N, C] ─┐
                                          ├→ Bidirectional Cross-Attn → Pool → MLP
        Mel → MViTv2 → spatial [B, M, C] ─┘
    """
    
    def __init__(self, num_classes=4, hidden_size=512, dropout_rate=0.2,
                 num_heads=8, num_cross_layers=2, fusion_type='concat'):
        super().__init__()
        
        # Spatial feature extractors
        self.maxvit = SpatialFeatureExtractor('maxvit_base_tf_224', pretrained=True)
        self.mvitv2 = SpatialFeatureExtractor('mvitv2_base', pretrained=True)
        
        # Get feature dimensions
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            spatial_cqt, pooled_cqt = self.maxvit(dummy)
            spatial_mel, pooled_mel = self.mvitv2(dummy)
            
            dim_cqt = spatial_cqt.shape[-1]
            dim_mel = spatial_mel.shape[-1]
            n_tokens_cqt = spatial_cqt.shape[1]
            n_tokens_mel = spatial_mel.shape[1]
            
        print(f"CQT: {n_tokens_cqt} tokens × {dim_cqt} dim")
        print(f"Mel: {n_tokens_mel} tokens × {dim_mel} dim")
        
        # Project to common dimension
        self.common_dim = max(dim_cqt, dim_mel)
        self.proj_cqt = nn.Linear(dim_cqt, self.common_dim) if dim_cqt != self.common_dim else nn.Identity()
        self.proj_mel = nn.Linear(dim_mel, self.common_dim) if dim_mel != self.common_dim else nn.Identity()
        
        # Bidirectional Cross-Attention
        self.cross_attention = BidirectionalCrossAttention(
            dim=self.common_dim,
            num_heads=num_heads,
            num_layers=num_cross_layers,
            mlp_ratio=4.,
            drop=dropout_rate,
            attn_drop=dropout_rate
        )
        
        # Fusion type
        self.fusion_type = fusion_type
        
        if fusion_type == 'concat':
            fusion_dim = self.common_dim * 2
        elif fusion_type == 'add':
            fusion_dim = self.common_dim
        elif fusion_type == 'gated':
            fusion_dim = self.common_dim
            self.gate = nn.Sequential(
                nn.Linear(self.common_dim * 2, self.common_dim),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
            
        # MLP Head
        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )
        
        print(f"Fusion type: {fusion_type}, output dim: {fusion_dim}")
        
    def forward(self, cqt, mel, return_attention=False):
        """
        Forward pass with proper cross-attention on spatial features.
        """
        # Expand to 3 channels
        if cqt.size(1) == 1:
            cqt = cqt.repeat(1, 3, 1, 1)
        if mel.size(1) == 1:
            mel = mel.repeat(1, 3, 1, 1)
            
        # Resize to 224x224
        if cqt.shape[-1] != 224:
            cqt = F.interpolate(cqt, size=(224, 224), mode='bilinear', align_corners=False)
        if mel.shape[-1] != 224:
            mel = F.interpolate(mel, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Extract SPATIAL features (before pooling)
        spatial_cqt, _ = self.maxvit(cqt)  # [B, N, dim_cqt]
        spatial_mel, _ = self.mvitv2(mel)  # [B, M, dim_mel]
        
        # Project to common dimension
        spatial_cqt = self.proj_cqt(spatial_cqt)  # [B, N, common_dim]
        spatial_mel = self.proj_mel(spatial_mel)  # [B, M, common_dim]
        
        # Bidirectional Cross-Attention
        feat_cqt, feat_mel, attn_info = self.cross_attention(spatial_cqt, spatial_mel)
        
        # Global Average Pooling
        pooled_cqt = feat_cqt.mean(dim=1)  # [B, common_dim]
        pooled_mel = feat_mel.mean(dim=1)  # [B, common_dim]
        
        # Fusion
        if self.fusion_type == 'concat':
            fused = torch.cat([pooled_cqt, pooled_mel], dim=1)
        elif self.fusion_type == 'add':
            fused = pooled_cqt + pooled_mel
        elif self.fusion_type == 'gated':
            concat = torch.cat([pooled_cqt, pooled_mel], dim=1)
            g = self.gate(concat)
            fused = g * pooled_cqt + (1 - g) * pooled_mel
            
        # Classification
        logits = self.mlp(fused)
        
        if return_attention:
            return logits, attn_info
        return logits


def get_optimizer_crossattn(model, lr=0.0002):
    """
    Optimizers:
    - Backbones (pretrained): lower lr
    - Cross-attention + MLP (new): higher lr
    """
    backbone_params = []
    new_params = []
    
    for name, param in model.named_parameters():
        if 'maxvit' in name or 'mvitv2' in name:
            backbone_params.append(param)
        else:
            new_params.append(param)
    
    # Backbone with lower lr
    opt_backbone = torch.optim.AdamW(backbone_params, lr=lr, weight_decay=0.01)
    
    # New layers with higher lr
    opt_new = torch.optim.AdamW(new_params, lr=lr * 5, weight_decay=0.01)
    
    return [opt_backbone, opt_new]


# ==================== Test ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing FIXED Cross-Attention MaxMViT-MLP")
    print("=" * 60)
    
    model = MaxMViT_MLP_CrossAttn(
        num_classes=4,
        num_heads=8,
        num_cross_layers=2,
        fusion_type='concat'
    )
    
    # Count params
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params: {total:,}")
    print(f"Trainable params: {trainable:,}")
    
    # Test
    print("\nTesting forward pass...")
    cqt = torch.randn(2, 1, 244, 244)
    mel = torch.randn(2, 1, 244, 244)
    
    logits, attn_info = model(cqt, mel, return_attention=True)
    
    print(f"Input: CQT {cqt.shape}, Mel {mel.shape}")
    print(f"Output: {logits.shape}")
    print(f"Attention layers: {len(attn_info['cqt_to_mel'])}")
    print(f"Attention shape (layer 0): {attn_info['cqt_to_mel'][0].shape}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
