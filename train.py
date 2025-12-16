
import argparse
import torch
import torch.nn as nn
import time
import os
import logging
from utils import load_config, setup_logging, seed_everything
from data_loaders import get_dataloaders
from model import MaxMViT_MLP, get_optimizer

def train(config_path):
    # 1. Load Config & Setup
    config = load_config(config_path)
    ckpt_dir = setup_logging(config)
    
    # 2. Extract Configs
    train_cfg = config['training']
    model_cfg = config['model']
    
    SEED = train_cfg.get('seed', 42)
    seed_everything(SEED)
    
    DEVICE = torch.device(train_cfg.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    EPOCHS = train_cfg.get('epochs', 50)
    LR = train_cfg.get('lr', 0.0002)
    PATIENCE = train_cfg.get('patience', 5)
    
    # 3. Data
    train_loader, val_loader = get_dataloaders(config)
    if not train_loader:
        logging.error("Failed to load data.")
        return

    # 4. Model
    num_classes = model_cfg.get('num_classes', 4)
    logging.info(f"Initializing Model with {num_classes} classes...")
    model = MaxMViT_MLP(num_classes=num_classes)
    model.to(DEVICE)
    
    # 5. Optimization
    optimizers = get_optimizer(model, lr=LR)
    
    sched_cfg = train_cfg.get('scheduler', {})
    schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', 
        factor=sched_cfg.get('factor', 0.1), 
        patience=sched_cfg.get('patience', 2), 
        min_lr=float(sched_cfg.get('min_lr', 1e-6))
    ) for opt in optimizers]
    
    criterion = nn.CrossEntropyLoss()
    
    # 6. Training Loop
    logging.info("Starting Training...")
    best_val_loss = float('inf')
    patience_counter = 0
    top_k_checkpoints = [] # {'loss': float, 'epoch': int, 'path': str}
    TOP_K = 3
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        start_time = time.time()
        
        for batch_idx, (cqt, mel, label) in enumerate(train_loader):
            cqt, mel, label = cqt.to(DEVICE), mel.to(DEVICE), label.to(DEVICE)
            
            for opt in optimizers: opt.zero_grad()
            
            outputs = model(cqt, mel)
            loss = criterion(outputs, label)
            loss.backward()
            
            # Clip Gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            for opt in optimizers: opt.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
            
            # Log every 10 batches (optional)
            if batch_idx % 20 == 0:
                 logging.debug(f"Batch {batch_idx}: Loss {loss.item():.4f}")

        # Epoch Metrics
        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation
        val_loss = 0
        val_correct = 0
        val_total = 0
        if val_loader:
            model.eval()
            with torch.no_grad():
                for cqt, mel, label in val_loader:
                    cqt, mel, label = cqt.to(DEVICE), mel.to(DEVICE), label.to(DEVICE)
                    outputs = model(cqt, mel)
                    loss = criterion(outputs, label)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += label.size(0)
                    val_correct += predicted.eq(label).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = 100. * val_correct / val_total
        else:
            val_loss = train_loss
            val_acc = train_acc

        # Step Scheduler
        for sch in schedulers: sch.step(val_loss)

        # Logging
        epoch_time = time.time() - start_time
        logging.info(f"Epoch {epoch+1:02d} | Train [L:{train_loss:.4f} A:{train_acc:.1f}%] | Val [L:{val_loss:.4f} A:{val_acc:.1f}%] | Time: {epoch_time:.1f}s")
        
        # Checkpointing Strategy (Top-K)
        filename = f"epoch_{epoch+1}.pth"
        save_path = os.path.join(ckpt_dir, filename)
        torch.save(model.state_dict(), save_path)
        
        # Maintain Top-K list
        top_k_checkpoints.append({'loss': val_loss, 'epoch': epoch+1, 'path': save_path})
        top_k_checkpoints.sort(key=lambda x: x['loss'])
        
        # Cleanup
        while len(top_k_checkpoints) > TOP_K:
            to_remove = top_k_checkpoints.pop() # Worst one
            if os.path.exists(to_remove['path']):
                os.remove(to_remove['path'])
                logging.info(f"Removed checkpoint: {os.path.basename(to_remove['path'])} (Loss: {to_remove['loss']:.4f})")
                
        # Early Stopping
        if top_k_checkpoints[0]['epoch'] == epoch + 1:
            patience_counter = 0 # New Best
            logging.info(f"New Best Model! Loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logging.info("Early stopping triggered.")
                break

    # Final Rename
    logging.info("Renaming Top Checkpoints...")
    for i, ckpt in enumerate(top_k_checkpoints):
        rank = i + 1
        new_name = f"rank{rank}_loss{ckpt['loss']:.4f}_epoch{ckpt['epoch']}.pth"
        new_path = os.path.join(ckpt_dir, new_name)
        if os.path.exists(ckpt['path']):
            os.rename(ckpt['path'], new_path)
            logging.info(f"Saved Rank {rank}: {new_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()
    
    train(args.config)
