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
    
    scheduler_cfg = training_cfg.get('scheduler', {})
    SCHEDULER_FACTOR = scheduler_cfg.get('factor', 0.1)
    SCHEDULER_PATIENCE = scheduler_cfg.get('patience', 2)
    MIN_LR = float(scheduler_cfg.get('min_lr', 1e-6))
    logging.info(f"Config loaded: {config}")
    
    # ... (Load Data block omitted for brevity in search, assuming it follows) ...
    # Initialize Early Stopping
    # Initialize Early Stopping and Top-K Saving
    best_val_loss = float('inf')
    patience_counter = 0
    top_k_checkpoints = [] # List of tuples: (loss, epoch, filename)
    TOP_K = 3
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
    schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, min_lr=MIN_LR
    ) for opt in optimizers]
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
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
            
            # Step Schedulers
            for sch in schedulers:
                sch.step(avg_test_loss)
            
 

            # Refined Implementation inside loop:
            # 1. Save current model directly if it *might* be top-k
            temp_filename = f"checkpoint_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), temp_filename)
            
            # 2. Add to list
            top_k_checkpoints.append({'loss': avg_test_loss, 'epoch': epoch+1, 'path': temp_filename})
            top_k_checkpoints.sort(key=lambda x: x['loss'])
            
            # 3. Remove extra (worst) checkpoints
            while len(top_k_checkpoints) > TOP_K:
                to_remove = top_k_checkpoints.pop() # Last one is worst
                if os.path.exists(to_remove['path']):
                    os.remove(to_remove['path'])
                    logging.info(f"   >>> Removed old checkpoint: {to_remove['path']}")

            # 4. Log current status
            best_val_loss = top_k_checkpoints[0]['loss'] # Update best_val_loss for Early Stopping
            current_top_losses = [x['loss'] for x in top_k_checkpoints]
            logging.info(f"   >>> Current Top {TOP_K} Losses: {current_top_losses}")
            
            # Early Stopping Check (based on best_val_loss)
            # If the BEST loss didn't change (i.e. we didn't add a new Rank 1), increment patience?
            # Or usually patience is "no improvement in BEST loss".
            if top_k_checkpoints[0]['epoch'] == epoch + 1:
                # We just found a new global best
                patience_counter = 0
                logging.info(f"   >>> Improvement! Patience reset.")
            else:
                patience_counter += 1
                logging.info(f"   >>> No improvement in best loss. Patience: {patience_counter}/{PATIENCE}")
                
            if patience_counter >= PATIENCE:
                logging.info("Early stopping triggered.")
                break
        else:
            # If no validation set, we just save last model
            logging.info("   >>> No validation set (HF dataset might not have validation split active).")
            torch.save(model.state_dict(), "last_model.pth")

    logging.info("Training Finished.")
    
    # Rename Top K Checkpoints
    logging.info("Renaming Top K Checkpoints...")
    for i, ckpt in enumerate(top_k_checkpoints):
        rank = i + 1
        old_path = ckpt['path']
        new_path = f"best_model_rank{rank}.pth"
        if os.path.exists(old_path):
            if os.path.exists(new_path):
                os.remove(new_path)
            os.rename(old_path, new_path)
            logging.info(f"   >>> Renamed {old_path} to {new_path} (Loss: {ckpt['loss']:.4f})")
