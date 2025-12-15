import torch
import torch.nn as nn
from model import MaxMViT_MLP, get_optimizer
from dataset_iemocap import get_iemocap_dataloaders
from dataset_hf import get_hf_dataloaders
import time
import os
import yaml
import logging

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log.txt"),
        logging.StreamHandler()
    ]
)

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    # Load Config
    config = load_config()
    
    dataset_cfg = config.get('dataset', {})
    training_cfg = config.get('training', {})
    model_cfg = config.get('model', {})
    
    # Settings
    EPOCHS = training_cfg.get('epochs', 10)
    BATCH_SIZE = dataset_cfg.get('batch_size', 4)
    NUM_WORKERS = dataset_cfg.get('num_workers', 4)
    NUM_CLASSES = model_cfg.get('num_classes', 4)
    
    device_str = training_cfg.get('device', 'cuda')
    DEVICE = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    
    ROOT_DIR = dataset_cfg.get('root_dir', "AbstractTTS/IEMOCAP")
    
    lr = training_cfg.get('lr', 0.02)
    PATIENCE = training_cfg.get('patience', 5)
    
    logging.info(f"Using device: {DEVICE}")
    logging.info(f"Config loaded: {config}")
    
    # ... (Load Data block omitted for brevity in search, assuming it follows) ...
    # Initialize Early Stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Load Data
    try:
        # Check if it looks like a HF ID (no path separators, usually Owner/Name)
        if "/" in ROOT_DIR and not os.path.exists(ROOT_DIR):
            logging.info("Detected Hugging Face Dataset ID. Loading from HF Hub...")
            train_loader, test_loader = get_hf_dataloaders(ROOT_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        else:
            logging.info("Loading from Local File System...")
            train_loader, test_loader = get_iemocap_dataloaders(ROOT_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
            
        if train_loader is None:
            raise ValueError("Failed to create dataloaders.")
            
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        # Only exit if critical, but for script usually yes
        exit()
        
    # Initialize Model
    model = MaxMViT_MLP(num_classes=NUM_CLASSES)
    model.to(DEVICE)
    
    # Optimizers
    optimizers = get_optimizer(model, lr=lr) 
    criterion = nn.CrossEntropyLoss()
    
    # Training Loop
    logging.info("Starting Training...")
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
        
        logging.info(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | Time: {time.time()-start_time:.2f}s")
        
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
            
            avg_test_loss = test_loss / len(test_loader)
            test_acc = 100. * test_correct / test_total if test_total > 0 else 0
            logging.info(f"   >>> Validation Loss: {avg_test_loss:.4f} | Acc: {test_acc:.2f}%")
            
            # Early Stopping Check
            if avg_test_loss < best_val_loss:
                best_val_loss = avg_test_loss
                patience_counter = 0
                torch.save(model.state_dict(), "best_model.pth")
                logging.info(f"   >>> Improved! Saved best_model.pth (Loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                logging.info(f"   >>> No improvement. Patience: {patience_counter}/{PATIENCE}")
                
            if patience_counter >= PATIENCE:
                logging.info("Early stopping triggered.")
                break
        else:
            # If no validation set, we just save last model
            logging.info("   >>> No validation set (HF dataset might not have validation split active).")
            torch.save(model.state_dict(), "last_model.pth")

    logging.info("Training Finished.")
