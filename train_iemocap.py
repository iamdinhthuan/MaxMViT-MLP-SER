import torch
import torch.nn as nn
from model import MaxMViT_MLP, get_optimizer
from dataset_iemocap import get_iemocap_dataloaders
from dataset_hf import get_hf_dataloaders
import time
import os

if __name__ == "__main__":
    # Settings
    EPOCHS = 10 
    BATCH_SIZE = 4 
    NUM_CLASSES = 4 # IEMOCAP 4 classes: neu, hap, ang, sad
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # User can set this to local path OR Hugging Face ID
    # ROOT_DIR = "mock_iemocap" 
    # ROOT_DIR = "d:/paper/ser/IEMOCAP_full_release"
    ROOT_DIR = "AbstractTTS/IEMOCAP" 
    
    print(f"Using device: {DEVICE}")
    print(f"Dataset root/ID: {ROOT_DIR}")
    
    # Load Data
    try:
        # Check if it looks like a HF ID (no path separators, usually Owner/Name)
        if "/" in ROOT_DIR and not os.path.exists(ROOT_DIR):
            print("Detected Hugging Face Dataset ID. Loading from HF Hub...")
            train_loader, test_loader = get_hf_dataloaders(ROOT_DIR, batch_size=BATCH_SIZE)
        else:
            print("Loading from Local File System...")
            train_loader, test_loader = get_iemocap_dataloaders(ROOT_DIR, batch_size=BATCH_SIZE)
            
        if train_loader is None:
            raise ValueError("Failed to create dataloaders.")
            
    except Exception as e:
        print(f"Error loading data: {e}")
        # Only exit if critical, but for script usually yes
        exit()
        
    # Initialize Model
    model = MaxMViT_MLP(num_classes=NUM_CLASSES)
    model.to(DEVICE)
    
    # Optimizers
    optimizers = get_optimizer(model, lr=0.001) 
    criterion = nn.CrossEntropyLoss()
    
    # Training Loop
    print("Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        start_time = time.time()
        
        # Add basic progress logging
        for batch_idx, (cqt, mel, label) in enumerate(train_loader):
            cqt, mel, label = cqt.to(DEVICE), mel.to(DEVICE), label.to(DEVICE)
            
            for opt in optimizers:
                opt.zero_grad()
            
            outputs = model(cqt, mel)
            loss = criterion(outputs, label)
            
            loss.backward()
            for opt in optimizers:
                opt.step()
                
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
            
            if batch_idx % 10 == 0:
                print(f"   Batch {batch_idx}: Loss {loss.item():.4f}")
            
        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        acc = 100.*correct/total if total > 0 else 0
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | Time: {time.time()-start_time:.2f}s")
        
        # Validation
        if test_loader:
            model.eval()
            test_loss = 0
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                 for cqt, mel, label in test_loader:
                    cqt, mel, label = cqt.to(DEVICE), mel.to(DEVICE), label.to(DEVICE)
                    outputs = model(cqt, mel)
                    loss = criterion(outputs, label)
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    test_total += label.size(0)
                    test_correct += predicted.eq(label).sum().item()
            
            test_acc = 100. * test_correct / test_total if test_total > 0 else 0
            print(f"   >>> Validation Acc: {test_acc:.2f}%")
        else:
            print("   >>> No validation set (HF dataset might not have validation split active).")

    print("\nTraining Finished.")
