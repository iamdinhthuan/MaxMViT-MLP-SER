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
        # Paper mentions MaxViT.        # Paper likely uses base/large. Switching to base as per feedback.
        # MaxViT Base
        self.maxvit = timm.create_model('maxvit_base_tf_224', pretrained=True, num_classes=0)
        # MViTv2 Base
        self.mvitv2 = timm.create_model('mvitv2_base', pretrained=True, num_classes=0)

        # Print config to verify window sizes if possible, or just the model name
        print(f"Initialized MaxViT: {self.maxvit.default_cfg['architecture']}")
        
        # Calculate feature dimension
        # We need to do a dummy forward pass or check config to know output dim.
        # Typically: MaxViT Small ~768, MViTv2 Small ~768 (checking needed)
        with torch.no_grad():
            # Use 224 for feature dim calc because we interpolate to 224 before backbone
            dummy_input = torch.randn(1, 3, 224, 224) 
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
        # Paper says 244x244. 
        # User requested 244x244.
        
        # MaxViT usually requires input divisible by 32 (224 is, 244 is NOT).
        # 244 / 32 = 7.625.
        # If we pass 244, MaxViT might error or perform poorly due to window padding.
        # However, to satisfy the requirement, we pass it through.
        # If strict compatibility is needed, we could interpolate to 224 here if we encounter errors.
        
        # Fix: MaxViT architecture restricts input size to be divisible by window size (7).
        # 244 is NOT divisible by 7. This causes a crash.
        # To support the user's request for 244 input (from config), we MUST interpolate to 224 
        # before the backbone to fit the fixed architecture constraints.
        if cqt.shape[-1] != 224:
             cqt = torch.nn.functional.interpolate(cqt, size=(224, 224), mode='bilinear', align_corners=False)
        if mel.shape[-1] != 224:
             mel = torch.nn.functional.interpolate(mel, size=(224, 224), mode='bilinear', align_corners=False)

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
