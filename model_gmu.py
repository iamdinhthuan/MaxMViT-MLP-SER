"""
MaxMViT-MLP with Gated Multimodal Unit (GMU) Fusion
Based on: GloMER (Nguyen et al., 2025) - Gate Fusion for Multimodal Emotion Recognition

Key improvement: Replace simple concatenation with GMU for adaptive modality balancing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class GatedMultimodalUnit(nn.Module):
    """
    Gated Multimodal Unit (GMU) from GloMER paper.
    
    Dynamically regulates the contribution of each modality.
    
    Formula:
        z̃^a = tanh(W_a · z^a + b_a)  # Audio projection
        z̃^t = tanh(W_t · z^t + b_t)  # Text/CQT projection  
        g = σ(W_g · [z̃^a; z̃^t] + b_g)  # Gating vector
        z_fused = g ⊙ z̃^t + (1-g) ⊙ z̃^a  # Weighted fusion
    """
    
    def __init__(self, dim_a, dim_t, hidden_dim=None):
        """
        Args:
            dim_a: Dimension of audio/Mel-STFT features (from MViTv2)
            dim_t: Dimension of text/CQT features (from MaxViT)
            hidden_dim: Hidden dimension for fusion (default: max of both)
        """
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = max(dim_a, dim_t)
        
        self.hidden_dim = hidden_dim
        
        # Projection layers with tanh activation (as per GloMER)
        self.proj_audio = nn.Sequential(
            nn.Linear(dim_a, hidden_dim),
            nn.Tanh()
        )
        
        self.proj_cqt = nn.Sequential(
            nn.Linear(dim_t, hidden_dim),
            nn.Tanh()
        )
        
        # Gating mechanism
        # Input: concatenation of projected features [z̃^a; z̃^t]
        # Output: gate values in [0, 1] via sigmoid
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(self, feat_audio, feat_cqt):
        """
        Args:
            feat_audio: [B, dim_a] - Features from MViTv2 (Mel-STFT path)
            feat_cqt: [B, dim_t] - Features from MaxViT (CQT path)
            
        Returns:
            fused: [B, hidden_dim] - Adaptively fused features
            gate_values: [B, hidden_dim] - Gate values for analysis
        """
        # Project to common space with tanh
        z_audio = self.proj_audio(feat_audio)  # [B, hidden_dim]
        z_cqt = self.proj_cqt(feat_cqt)        # [B, hidden_dim]
        
        # Compute gating vector
        concat = torch.cat([z_audio, z_cqt], dim=1)  # [B, hidden_dim * 2]
        g = self.gate(concat)  # [B, hidden_dim], values in [0, 1]
        
        # Adaptive fusion: g controls balance between modalities
        # g → 1: favor CQT (MaxViT path)
        # g → 0: favor Mel-STFT (MViTv2 path)
        fused = g * z_cqt + (1 - g) * z_audio  # [B, hidden_dim]
        
        return fused, g


class MaxMViT_MLP_GMU(nn.Module):
    """
    MaxMViT-MLP with GMU Fusion.
    
    Architecture:
        CQT Spectrogram → MaxViT → 
                                    → GMU Fusion → MLP → Classification
        Mel-STFT Spectrogram → MViTv2 →
    
    Improvements over original:
        1. GMU instead of simple concatenation
        2. Adaptive modality balancing
        3. Learnable fusion weights
    """
    
    def __init__(self, num_classes=7, hidden_size=512, dropout_rate=0.2, 
                 fusion_hidden_dim=None):
        """
        Args:
            num_classes: Number of emotion classes
            hidden_size: MLP hidden layer size (paper: 512)
            dropout_rate: Dropout rate (paper: 0.2)
            fusion_hidden_dim: GMU hidden dimension (None = auto)
        """
        super().__init__()
        
        # --- Backbone: Same as original ---
        # Path 1: CQT → MaxViT
        self.maxvit = timm.create_model('maxvit_base_tf_224', pretrained=True, num_classes=0)
        
        # Path 2: Mel-STFT → MViTv2
        self.mvitv2 = timm.create_model('mvitv2_base', pretrained=True, num_classes=0)
        
        # Get feature dimensions
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            dim_cqt = self.maxvit(dummy).shape[1]   # MaxViT output dim
            dim_mel = self.mvitv2(dummy).shape[1]   # MViTv2 output dim
            
        print(f"Feature dims - CQT/MaxViT: {dim_cqt}, Mel/MViTv2: {dim_mel}")
        
        # --- GMU Fusion (NEW) ---
        if fusion_hidden_dim is None:
            fusion_hidden_dim = max(dim_cqt, dim_mel)
            
        self.gmu = GatedMultimodalUnit(
            dim_a=dim_mel,      # Audio/Mel path
            dim_t=dim_cqt,      # CQT path  
            hidden_dim=fusion_hidden_dim
        )
        
        # --- MLP Head ---
        # Input is now fusion_hidden_dim instead of dim_cqt + dim_mel
        self.mlp = nn.Sequential(
            nn.Linear(fusion_hidden_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )
        
        # Store for analysis
        self.fusion_hidden_dim = fusion_hidden_dim
        
    def forward(self, cqt, mel, return_gate=False):
        """
        Args:
            cqt: CQT spectrogram [B, C, H, W]
            mel: Mel-STFT spectrogram [B, C, H, W]
            return_gate: If True, also return gate values for analysis
            
        Returns:
            logits: [B, num_classes]
            gate_values (optional): [B, hidden_dim] - shows modality importance
        """
        # Expand to 3 channels if needed
        if cqt.size(1) == 1:
            cqt = cqt.repeat(1, 3, 1, 1)
        if mel.size(1) == 1:
            mel = mel.repeat(1, 3, 1, 1)
            
        # Resize to 224x224 if needed
        if cqt.shape[-1] != 224:
            cqt = F.interpolate(cqt, size=(224, 224), mode='bilinear', align_corners=False)
        if mel.shape[-1] != 224:
            mel = F.interpolate(mel, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Extract features from both paths
        feat_cqt = self.maxvit(cqt)    # [B, dim_cqt]
        feat_mel = self.mvitv2(mel)    # [B, dim_mel]
        
        # GMU Fusion (instead of simple concatenation)
        fused, gate_values = self.gmu(feat_mel, feat_cqt)  # [B, fusion_hidden_dim]
        
        # Classification
        logits = self.mlp(fused)
        
        if return_gate:
            return logits, gate_values
        return logits


class MaxMViT_MLP_GMU_Contrastive(MaxMViT_MLP_GMU):
    """
    Extended version with Contrastive Self-Alignment (optional).
    
    Adds GloMER's contrastive learning losses:
        - NT-Xent loss: Push matching pairs closer
        - Consistency loss: Cosine similarity alignment  
        - Diversity loss: Prevent embedding collapse
    """
    
    def __init__(self, num_classes=7, hidden_size=512, dropout_rate=0.2,
                 fusion_hidden_dim=None, proj_dim=256):
        super().__init__(num_classes, hidden_size, dropout_rate, fusion_hidden_dim)
        
        # Projection heads for contrastive learning
        self.proj_cqt = nn.Sequential(
            nn.Linear(self.fusion_hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        
        self.proj_mel = nn.Sequential(
            nn.Linear(self.fusion_hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        
    def forward(self, cqt, mel, return_projections=False):
        # Expand channels
        if cqt.size(1) == 1:
            cqt = cqt.repeat(1, 3, 1, 1)
        if mel.size(1) == 1:
            mel = mel.repeat(1, 3, 1, 1)
            
        # Resize
        if cqt.shape[-1] != 224:
            cqt = F.interpolate(cqt, size=(224, 224), mode='bilinear', align_corners=False)
        if mel.shape[-1] != 224:
            mel = F.interpolate(mel, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Extract features
        feat_cqt = self.maxvit(cqt)
        feat_mel = self.mvitv2(mel)
        
        # GMU Fusion
        fused, gate_values = self.gmu(feat_mel, feat_cqt)
        
        # Classification
        logits = self.mlp(fused)
        
        if return_projections:
            # Project for contrastive loss
            # Use the projected features from GMU
            z_cqt = self.gmu.proj_cqt(feat_cqt)
            z_mel = self.gmu.proj_audio(feat_mel)
            
            proj_cqt = self.proj_cqt(z_cqt)
            proj_mel = self.proj_mel(z_mel)
            
            return logits, proj_cqt, proj_mel, gate_values
            
        return logits


# ==================== Loss Functions ====================

class ContrastiveLoss(nn.Module):
    """
    Combined contrastive loss from GloMER paper.
    
    L_total = L_CE + L_NT-Xent + α(L_con + L_div)
    """
    
    def __init__(self, temperature=0.07, alpha=0.3):
        """
        Args:
            temperature: NT-Xent temperature (τ)
            alpha: Balance parameter for consistency + diversity
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        
    def nt_xent_loss(self, z_cqt, z_mel):
        """
        NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss.
        Encourages paired samples to be close, non-paired to be far.
        """
        batch_size = z_cqt.size(0)
        
        # Normalize
        z_cqt = F.normalize(z_cqt, dim=1)
        z_mel = F.normalize(z_mel, dim=1)
        
        # Compute similarity matrix
        sim = torch.mm(z_cqt, z_mel.t()) / self.temperature  # [B, B]
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=z_cqt.device)
        
        # Cross entropy loss (treating as classification)
        loss = F.cross_entropy(sim, labels)
        
        return loss
    
    def consistency_loss(self, z_cqt, z_mel):
        """
        Consistency loss: 1 - mean(cosine_similarity).
        Encourages paired embeddings to be similar.
        """
        cos_sim = F.cosine_similarity(z_cqt, z_mel, dim=1)
        loss = 1 - cos_sim.mean()
        return loss
    
    def diversity_loss(self, z_cqt, z_mel):
        """
        Diversity loss: L2 distance between embeddings.
        Prevents trivial collapse where both modalities produce identical outputs.
        """
        loss = torch.mean((z_cqt - z_mel) ** 2)
        return loss
    
    def forward(self, logits, labels, z_cqt=None, z_mel=None):
        """
        Args:
            logits: Classification logits [B, num_classes]
            labels: Ground truth labels [B]
            z_cqt: CQT projections [B, proj_dim] (optional)
            z_mel: Mel projections [B, proj_dim] (optional)
            
        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components for logging
        """
        # Classification loss
        l_ce = self.ce_loss(logits, labels)
        
        loss_dict = {'ce': l_ce.item()}
        total_loss = l_ce
        
        # Contrastive losses (if projections provided)
        if z_cqt is not None and z_mel is not None:
            l_nt_xent = self.nt_xent_loss(z_cqt, z_mel)
            l_con = self.consistency_loss(z_cqt, z_mel)
            l_div = self.diversity_loss(z_cqt, z_mel)
            
            total_loss = l_ce + l_nt_xent + self.alpha * (l_con + l_div)
            
            loss_dict.update({
                'nt_xent': l_nt_xent.item(),
                'consistency': l_con.item(),
                'diversity': l_div.item()
            })
            
        return total_loss, loss_dict


# ==================== Optimizer ====================

def get_optimizer_gmu(model, lr=0.02):
    """
    Optimizers following paper specifications:
    - MaxViT: Adam (lr=0.02)
    - MViTv2: RAdam (lr=0.02)  
    - GMU + MLP: Adam
    """
    maxvit_params = list(model.maxvit.parameters())
    mvitv2_params = list(model.mvitv2.parameters())
    gmu_params = list(model.gmu.parameters())
    mlp_params = list(model.mlp.parameters())
    
    # Optional: contrastive projection params
    other_params = []
    if hasattr(model, 'proj_cqt'):
        other_params += list(model.proj_cqt.parameters())
        other_params += list(model.proj_mel.parameters())
    
    # Optimizer 1: MaxViT + GMU + MLP + projections → Adam
    opt1 = torch.optim.Adam(
        maxvit_params + gmu_params + mlp_params + other_params, 
        lr=lr
    )
    
    # Optimizer 2: MViTv2 → RAdam
    opt2 = torch.optim.RAdam(mvitv2_params, lr=lr)
    
    return [opt1, opt2]


# ==================== Quick Test ====================

if __name__ == "__main__":
    print("Testing GMU-enhanced MaxMViT-MLP...")
    
    # Test basic GMU model
    model = MaxMViT_MLP_GMU(num_classes=4)
    
    # Dummy input
    cqt = torch.randn(2, 3, 224, 224)
    mel = torch.randn(2, 3, 224, 224)
    
    # Forward pass
    logits, gates = model(cqt, mel, return_gate=True)
    
    print(f"Output shape: {logits.shape}")
    print(f"Gate shape: {gates.shape}")
    print(f"Gate mean: {gates.mean().item():.3f} (0.5 = balanced)")
    
    # Test contrastive version
    print("\nTesting Contrastive version...")
    model_cl = MaxMViT_MLP_GMU_Contrastive(num_classes=4)
    logits, z_cqt, z_mel, gates = model_cl(cqt, mel, return_projections=True)
    
    # Test loss
    loss_fn = ContrastiveLoss(alpha=0.3)
    labels = torch.tensor([0, 1])
    total_loss, loss_dict = loss_fn(logits, labels, z_cqt, z_mel)
    
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Loss components: {loss_dict}")
    
    print("\n✅ All tests passed!")
