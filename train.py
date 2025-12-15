import torch
import torch.nn as nn
from model import MaxMViT_MLP, get_optimizer
from dataset import get_dataloader
import os

# Example Usage
if __name__ == "__main__":
    # Settings
    EPOCHS = 50
    BATCH_SIZE = 16 # Adjust based on GPU memory
    NUM_CLASSES = 7 # Emo-DB: 7
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {DEVICE}")
    
    # Placeholder Data (Replace with actual data loading)
    # create dummy wav files if not exist for testing purposes
    if not os.path.exists("test_audio.wav"):
        import soundfile as sf
        import numpy as np
        sr = 44100
        dummy_audio = np.random.uniform(-1, 1, sr * 3) # 3 seconds
        sf.write('test_audio.wav', dummy_audio, sr)
        
    paths = ["test_audio.wav"] * 10
    labels = [0] * 10
    
    dataloader = get_dataloader(paths, labels, batch_size=BATCH_SIZE)
    
    # Initialize Model
    model = MaxMViT_MLP(num_classes=NUM_CLASSES)
    model.to(DEVICE)
    
    # Get Optimizers
    # Optimizer 1: Adam (MaxViT), Optimizer 2: RAdam (MViTv2)
    optimizers = get_optimizer(model, lr=0.02)
    
    criterion = nn.CrossEntropyLoss()
    
    # Training Loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (cqt, mel, label) in enumerate(dataloader):
            cqt, mel, label = cqt.to(DEVICE), mel.to(DEVICE), label.to(DEVICE)
            
            # Zero gradients for all optimizers
            for opt in optimizers:
                opt.zero_grad()
            
            # Forward
            outputs = model(cqt, mel)
            loss = criterion(outputs, label)
            
            # Backward
            loss.backward()
            
            # Step all optimizers
            for opt in optimizers:
                opt.step()
                
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dataloader):.4f} | Acc: {100.*correct/total:.2f}%")
        
    print("Training Complete")
