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
        # Using 'maxvit_rmlp_base_rw_224' or similar. 
        # Paper mentions MaxViT. We use a standard variant.
        # We need to remove the classifier head to get features.
        self.maxvit = timm.create_model('maxvit_rmlp_small_rw_224', pretrained=True, num_classes=0) 
        
        # --- Path 2: Mel-STFT + MViTv2 ---
        # Using 'mvitv2_small' or similar.
        self.mvitv2 = timm.create_model('mvitv2_small', pretrained=True, num_classes=0)
        
        # Calculate feature dimension
        # We need to do a dummy forward pass or check config to know output dim.
        # Typically: MaxViT Small ~768, MViTv2 Small ~768 (checking needed)
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224) # Standard size
            maxvit_dim = self.maxvit(dummy_input).shape[1]
            mvitv2_dim = self.mvitv2(dummy_input).shape[1]
            
        fusion_dim = maxvit_dim + mvitv2_dim
        
        # --- MLP Head ---
        # Dense layer -> Batch Norm -> Dropout -> Classification
        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout_rate),
            nn.ReLU(), # Paper implies activation before classification? 
                       # "Dense layer... followed by classification layer... softmax"
                       # Usually Dense -> Activation -> BN -> Dropout -> FC.
                       # Paper text: "dense layer, batch normalization layer, dropout layer, and a classification layer."
                       # "two dense neural network layers activated by the ReLU function" (Ref to Vu et al. [13], not this work?)
                       # Section III.D.1: "Dense layer... applies linear transformation... BN... Dropout... Classification layer computes probabilities... softmax"
                       # Usually Linear implies just linear. But networks need non-linearity.
                       # I will add ReLU for safety as "Dense Layer" typically implies a hidden layer with activation.
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, cqt, mel):
        """
        Forward pass.
        
        Args:
           cqt (torch.Tensor): CQT Spectrogram [Batch, 1, 244, 244] -> will repeat to 3 channels
           mel (torch.Tensor): Mel-STFT Spectrogram [Batch, 1, 244, 244]
        """
        # Expand 1 channel to 3 channels for backbone compatibility
        if cqt.size(1) == 1:
            cqt = cqt.repeat(1, 3, 1, 1)
        if mel.size(1) == 1:
            mel = mel.repeat(1, 3, 1, 1)
            
        # Resize to 224x224 if model expects it (timm models usually strict or better at native res)
        # Paper says 244x244. MaxViT supports arbitrary. MViTv2 might. 
        # Using interpolation to match standard 224 if needed, but let's try keeping input size.
        # Actually input is 244, let's interpolate to 224 for 'safe' pretrained usage 
        # unless we want to handle positional embedding interpolation.
        # timm handles it usually, but let's be safe.
        # Paper uses 244x244. timm models usually interpolate pos embeddings automatically.
        # We pass the 244x244 input directly.
        pass

        # Path 1
        feat_maxvit = self.maxvit(cqt) # [B, Dim1]
        
        # Path 2
        feat_mvitv2 = self.mvitv2(mel) # [B, Dim2]
        
        # Fusion
        fused = torch.cat((feat_maxvit, feat_mvitv2), dim=1)
        
        # MLP
        logits = self.mlp(fused)
        
        return logits

def get_optimizer(model, lr=0.02):
    """
    Returns the optimizers as specified in the paper:
    - MaxViT: Adam
    - MViTv2: RAdam
    - MLP: Assuming Adam (matches MaxViT or dominant)
    """
    # Split parameters
    maxvit_params = list(model.maxvit.parameters())
    mvitv2_params = list(model.mvitv2.parameters())
    mlp_params = list(model.mlp.parameters())
    
    # Optimizer 1: MaxViT (and let's put MLP here too) -> Adam
    optimizer1 = torch.optim.Adam(maxvit_params + mlp_params, lr=lr)
    
    # Optimizer 2: MViTv2 -> RAdam
    optimizer2 = torch.optim.RAdam(mvitv2_params, lr=lr)
    
    return [optimizer1, optimizer2] 
