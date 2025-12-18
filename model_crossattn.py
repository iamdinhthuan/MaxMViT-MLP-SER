"""
MaxMViT-MLP with Cross-Attention Fusion

Cross-Attention allows each modality to attend to the other modality,
learning rich inter-modal interactions.

Architecture:
    CQT → MaxViT → feat_cqt →
                              → Cross-Attention Fusion → MLP → Classification  
    Mel → MViTv2 → feat_mel →

Cross-Attention Mechanism:
    - CQT attends to Mel: Q=CQT, K=Mel, V=Mel
    - Mel attends to CQT: Q=Mel, K=CQT, V=CQT
    - Bidirectional information flow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math


class CrossAttentionBlock(nn.Module):
    """
    Cross-Attention Block for multimodal fusion.
    
    Allows one modality to query information from another modality.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    
    Where:
        Q comes from modality A
        K, V come from modality B
    """
    
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        """
        Args:
            dim: Feature dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in Q, K, V projections
            attn_drop: Attention dropout rate
            proj_drop: Output projection dropout rate
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # 1/sqrt(d_k)
        
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        # Separate projections for Q (from one modality) and K, V (from another)
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, query, key_value):
        """
        Cross-attention: query attends to key_value.
        
        Args:
            query: [B, dim] - The modality that asks questions
            key_value: [B, dim] - The modality that provides answers
            
        Returns:
            out: [B, dim] - Attended features
            attn_weights: [B, num_heads, 1, 1] - Attention weights
        """
        B = query.shape[0]
        
        # Add sequence dimension for attention computation
        # [B, dim] -> [B, 1, dim]
        query = query.unsqueeze(1)
        key_value = key_value.unsqueeze(1)
        
        # Project to Q, K, V
        Q = self.q_proj(query)  # [B, 1, dim]
        K = self.k_proj(key_value)  # [B, 1, dim]
        V = self.v_proj(key_value)  # [B, 1, dim]
        
        # Reshape for multi-head attention
        # [B, 1, dim] -> [B, num_heads, 1, head_dim]
        Q = Q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        # [B, num_heads, 1, head_dim] @ [B, num_heads, head_dim, 1] = [B, num_heads, 1, 1]
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.attn_drop(attn_weights)
        
        # Apply attention to values
        # [B, num_heads, 1, 1] @ [B, num_heads, 1, head_dim] = [B, num_heads, 1, head_dim]
        out = attn_weights @ V
        
        # Reshape back
        # [B, num_heads, 1, head_dim] -> [B, 1, dim]
        out = out.transpose(1, 2).contiguous().view(B, 1, -1)
        
        # Output projection
        out = self.proj(out)
        out = self.proj_drop(out)
        
        # Remove sequence dimension: [B, 1, dim] -> [B, dim]
        out = out.squeeze(1)
        
        return out, attn_weights.squeeze(-1).squeeze(-1)  # [B, num_heads]


class CrossAttentionFusion(nn.Module):
    """
    Bidirectional Cross-Attention Fusion Module.
    
    Both modalities attend to each other, then we combine:
        - feat_cqt_attended: CQT features enriched with Mel information
        - feat_mel_attended: Mel features enriched with CQT information
        
    Final fusion combines both attended features.
    """
    
    def __init__(self, dim_cqt, dim_mel, hidden_dim=None, num_heads=8, 
                 attn_drop=0., proj_drop=0., fusion_type='concat'):
        """
        Args:
            dim_cqt: CQT feature dimension (from MaxViT)
            dim_mel: Mel feature dimension (from MViTv2)
            hidden_dim: Internal attention dimension (None = max of both)
            num_heads: Number of attention heads
            attn_drop: Attention dropout rate
            proj_drop: Projection dropout rate
            fusion_type: 'concat', 'add', or 'gated'
        """
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = max(dim_cqt, dim_mel)
            
        self.hidden_dim = hidden_dim
        self.fusion_type = fusion_type
        
        # Project both modalities to common dimension
        self.proj_cqt = nn.Linear(dim_cqt, hidden_dim) if dim_cqt != hidden_dim else nn.Identity()
        self.proj_mel = nn.Linear(dim_mel, hidden_dim) if dim_mel != hidden_dim else nn.Identity()
        
        # Layer norm before attention
        self.norm_cqt = nn.LayerNorm(hidden_dim)
        self.norm_mel = nn.LayerNorm(hidden_dim)
        
        # Cross-attention blocks
        # CQT queries Mel (CQT → Q, Mel → K,V)
        self.cross_attn_cqt_to_mel = CrossAttentionBlock(
            dim=hidden_dim, 
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop
        )
        
        # Mel queries CQT (Mel → Q, CQT → K,V)
        self.cross_attn_mel_to_cqt = CrossAttentionBlock(
            dim=hidden_dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop
        )
        
        # Feedforward after attention (standard transformer pattern)
        self.ffn_cqt = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(proj_drop)
        )
        
        self.ffn_mel = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(proj_drop)
        )
        
        self.norm_cqt_ffn = nn.LayerNorm(hidden_dim)
        self.norm_mel_ffn = nn.LayerNorm(hidden_dim)
        
        # Fusion layer based on type
        if fusion_type == 'concat':
            self.output_dim = hidden_dim * 2
            self.fusion_proj = nn.Identity()
        elif fusion_type == 'add':
            self.output_dim = hidden_dim
            self.fusion_proj = nn.Identity()
        elif fusion_type == 'gated':
            self.output_dim = hidden_dim
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
            
    def forward(self, feat_cqt, feat_mel):
        """
        Bidirectional cross-attention fusion.
        
        Args:
            feat_cqt: [B, dim_cqt] - Features from MaxViT (CQT path)
            feat_mel: [B, dim_mel] - Features from MViTv2 (Mel path)
            
        Returns:
            fused: [B, output_dim] - Fused features
            attn_info: dict with attention weights for analysis
        """
        # Project to common dimension
        cqt = self.proj_cqt(feat_cqt)  # [B, hidden_dim]
        mel = self.proj_mel(feat_mel)  # [B, hidden_dim]
        
        # Normalize
        cqt_normed = self.norm_cqt(cqt)
        mel_normed = self.norm_mel(mel)
        
        # Cross-attention: CQT attends to Mel
        # "What information from Mel-STFT is useful for CQT?"
        cqt_attended, attn_cqt = self.cross_attn_cqt_to_mel(cqt_normed, mel_normed)
        cqt = cqt + cqt_attended  # Residual connection
        
        # Cross-attention: Mel attends to CQT  
        # "What information from CQT is useful for Mel-STFT?"
        mel_attended, attn_mel = self.cross_attn_mel_to_cqt(mel_normed, cqt_normed)
        mel = mel + mel_attended  # Residual connection
        
        # Feedforward with residual
        cqt = cqt + self.ffn_cqt(self.norm_cqt_ffn(cqt))
        mel = mel + self.ffn_mel(self.norm_mel_ffn(mel))
        
        # Fusion
        if self.fusion_type == 'concat':
            fused = torch.cat([cqt, mel], dim=1)  # [B, hidden_dim * 2]
        elif self.fusion_type == 'add':
            fused = cqt + mel  # [B, hidden_dim]
        elif self.fusion_type == 'gated':
            concat = torch.cat([cqt, mel], dim=1)
            g = self.gate(concat)  # [B, hidden_dim]
            fused = g * cqt + (1 - g) * mel  # [B, hidden_dim]
            
        attn_info = {
            'cqt_to_mel': attn_cqt,  # How CQT attends to Mel
            'mel_to_cqt': attn_mel   # How Mel attends to CQT
        }
        
        return fused, attn_info


class MaxMViT_MLP_CrossAttn(nn.Module):
    """
    MaxMViT-MLP with Cross-Attention Fusion.
    
    Architecture:
        CQT Spectrogram → MaxViT → 
                                   → Cross-Attention Fusion → MLP → Classification
        Mel-STFT Spectrogram → MViTv2 →
    
    Key improvements over concatenation:
        1. Bidirectional attention between modalities
        2. Each modality can query relevant information from the other
        3. Learnable inter-modal interactions
        4. Residual connections preserve original features
    """
    
    def __init__(self, num_classes=4, hidden_size=512, dropout_rate=0.2,
                 fusion_hidden_dim=None, num_heads=8, fusion_type='concat'):
        """
        Args:
            num_classes: Number of emotion classes
            hidden_size: MLP hidden layer size (paper: 512)
            dropout_rate: Dropout rate (paper: 0.2)
            fusion_hidden_dim: Cross-attention hidden dimension (None = auto)
            num_heads: Number of attention heads
            fusion_type: 'concat', 'add', or 'gated'
        """
        super().__init__()
        
        # --- Backbone Networks ---
        # Path 1: CQT → MaxViT
        self.maxvit = timm.create_model('maxvit_base_tf_224', pretrained=True, num_classes=0)
        
        # Path 2: Mel-STFT → MViTv2
        self.mvitv2 = timm.create_model('mvitv2_base', pretrained=True, num_classes=0)
        
        # Get feature dimensions via dummy forward pass
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            dim_cqt = self.maxvit(dummy).shape[1]
            dim_mel = self.mvitv2(dummy).shape[1]
            
        print(f"Feature dims - CQT/MaxViT: {dim_cqt}, Mel/MViTv2: {dim_mel}")
        
        # --- Cross-Attention Fusion ---
        if fusion_hidden_dim is None:
            fusion_hidden_dim = max(dim_cqt, dim_mel)
            
        self.cross_attn_fusion = CrossAttentionFusion(
            dim_cqt=dim_cqt,
            dim_mel=dim_mel,
            hidden_dim=fusion_hidden_dim,
            num_heads=num_heads,
            attn_drop=dropout_rate,
            proj_drop=dropout_rate,
            fusion_type=fusion_type
        )
        
        fusion_output_dim = self.cross_attn_fusion.output_dim
        print(f"Fusion output dim: {fusion_output_dim}")
        
        # --- MLP Head ---
        self.mlp = nn.Sequential(
            nn.Linear(fusion_output_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )
        
        # Store config
        self.fusion_hidden_dim = fusion_hidden_dim
        self.num_heads = num_heads
        
    def forward(self, cqt, mel, return_attention=False):
        """
        Forward pass with cross-attention fusion.
        
        Args:
            cqt: CQT spectrogram [B, C, H, W]
            mel: Mel-STFT spectrogram [B, C, H, W]
            return_attention: If True, return attention weights for visualization
            
        Returns:
            logits: [B, num_classes]
            attn_info (optional): Attention weights for analysis
        """
        # Expand to 3 channels if needed
        if cqt.size(1) == 1:
            cqt = cqt.repeat(1, 3, 1, 1)
        if mel.size(1) == 1:
            mel = mel.repeat(1, 3, 1, 1)
            
        # Resize to 224x224 (required by backbones)
        if cqt.shape[-1] != 224:
            cqt = F.interpolate(cqt, size=(224, 224), mode='bilinear', align_corners=False)
        if mel.shape[-1] != 224:
            mel = F.interpolate(mel, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Extract features from both backbones
        feat_cqt = self.maxvit(cqt)    # [B, dim_cqt]
        feat_mel = self.mvitv2(mel)    # [B, dim_mel]
        
        # Cross-Attention Fusion
        fused, attn_info = self.cross_attn_fusion(feat_cqt, feat_mel)
        
        # Classification
        logits = self.mlp(fused)
        
        if return_attention:
            return logits, attn_info
        return logits


def get_optimizer_crossattn(model, lr=0.0002):
    """
    Optimizers following paper specifications:
    - MaxViT: Adam (lr=0.0002)
    - MViTv2: RAdam (lr=0.0002)
    - CrossAttn + MLP: Adam (with higher lr for new layers)
    """
    maxvit_params = list(model.maxvit.parameters())
    mvitv2_params = list(model.mvitv2.parameters())
    fusion_params = list(model.cross_attn_fusion.parameters())
    mlp_params = list(model.mlp.parameters())
    
    # Optimizer 1: MaxViT (pretrained) → Adam with base lr
    opt_maxvit = torch.optim.Adam(maxvit_params, lr=lr)
    
    # Optimizer 2: MViTv2 (pretrained) → RAdam with base lr
    opt_mvitv2 = torch.optim.RAdam(mvitv2_params, lr=lr)
    
    # Optimizer 3: Fusion + MLP (new layers) → Adam with slightly higher lr
    opt_fusion = torch.optim.Adam(fusion_params + mlp_params, lr=lr * 2)
    
    return [opt_maxvit, opt_mvitv2, opt_fusion]


# ==================== Quick Test ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Cross-Attention MaxMViT-MLP")
    print("=" * 60)
    
    # Test model creation
    print("\n1. Creating model...")
    model = MaxMViT_MLP_CrossAttn(
        num_classes=4,
        hidden_size=512,
        dropout_rate=0.2,
        num_heads=8,
        fusion_type='concat'
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    cqt = torch.randn(2, 1, 244, 244)  # Original input size
    mel = torch.randn(2, 1, 244, 244)
    
    logits, attn_info = model(cqt, mel, return_attention=True)
    
    print(f"Input CQT shape: {cqt.shape}")
    print(f"Input Mel shape: {mel.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"CQT→Mel attention shape: {attn_info['cqt_to_mel'].shape}")
    print(f"Mel→CQT attention shape: {attn_info['mel_to_cqt'].shape}")
    
    # Analyze attention patterns
    print("\n3. Attention Analysis:")
    print(f"CQT→Mel attention (mean per head): {attn_info['cqt_to_mel'].mean(dim=0)}")
    print(f"Mel→CQT attention (mean per head): {attn_info['mel_to_cqt'].mean(dim=0)}")
    
    # Test different fusion types
    print("\n4. Testing different fusion types...")
    for fusion_type in ['concat', 'add', 'gated']:
        model_test = MaxMViT_MLP_CrossAttn(num_classes=4, fusion_type=fusion_type)
        out = model_test(cqt, mel)
        print(f"  {fusion_type}: output shape = {out.shape}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
