import torch
import torch.nn as nn
import timm
import torchaudio
import librosa
import numpy as np
import cv2
import os

class MaxMViT_MLP(nn.Module):
    def __init__(self, num_classes=7, hidden_size=512, dropout_rate=0.2):
        """
        MaxMViT and MViTv2 Fusion Network with Multilayer Perceptron (MaxMViT-MLP).
        
        Args:
            num_classes (int): Number of emotion classes (e.g., 7 for Emo-DB).
            hidden_size (int): Number of hidden nodes in MLP (default 512).
            dropout_rate (float): Dropout rate (default 0.2).
        """
        super(MaxMViT_MLP, self).__init__()
        
        # --- Path 1: CQT + MaxViT ---
        self.maxvit = timm.create_model('maxvit_base_tf_224', pretrained=True, num_classes=0)
        # --- Path 2: MEL + MViTv2 ---
        self.mvitv2 = timm.create_model('mvitv2_base', pretrained=True, num_classes=0)

        print(f"Initialized MaxViT: {self.maxvit.default_cfg['architecture']}")
        
        # Calculate feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224) 
            maxvit_dim = self.maxvit(dummy_input).shape[1]
            mvitv2_dim = self.mvitv2(dummy_input).shape[1]
            
        fusion_dim = maxvit_dim + mvitv2_dim
        
        # --- MLP Head ---
        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

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
        
        # Fusion: Simple Concatenation
        fused = torch.cat((feat_maxvit, feat_mvitv2), dim=1)  # [B, 1536]
        
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
    
    # Optimizer 1: MaxViT + MLP -> Adam
    optimizer1 = torch.optim.Adam(maxvit_params + mlp_params, lr=lr)
    
    # Optimizer 2: MViTv2 -> RAdam
    optimizer2 = torch.optim.RAdam(mvitv2_params, lr=lr)
    
    return [optimizer1, optimizer2]
