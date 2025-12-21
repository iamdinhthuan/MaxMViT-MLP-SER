"""
Train and Compare All Model Variants

This script trains all 3 fusion methods:
1. Original (Simple Concatenation)
2. GMU (Gated Multimodal Unit)
3. CrossAttn (Bidirectional Cross-Attention)

And generates a comparison table at the end.
"""

import argparse
import torch
import torch.nn as nn
import time
import os
import logging
import warnings
import json
from datetime import datetime
from tabulate import tabulate
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress librosa n_fft warnings
warnings.filterwarnings('ignore', message='n_fft=.*is too large for input signal')

from utils import load_config, setup_logging, seed_everything
from data_loaders import get_dataloaders

# Model imports
from model import MaxMViT_MLP, get_optimizer
from model_gmu import MaxMViT_MLP_GMU, get_optimizer_gmu
from model_crossattn import MaxMViT_MLP_CrossAttn, get_optimizer_crossattn


def get_model_and_optimizer(model_type, num_classes, lr, model_cfg):
    """Factory function to get model and optimizer based on model_type."""
    hidden_size = model_cfg.get('hidden_size', 512)
    dropout_rate = model_cfg.get('dropout_rate', 0.2)
    
    if model_type == 'original':
        model = MaxMViT_MLP(num_classes=num_classes, hidden_size=hidden_size, dropout_rate=dropout_rate)
        optimizers = get_optimizer(model, lr=lr)
        
    elif model_type == 'gmu':
        fusion_hidden_dim = model_cfg.get('fusion_hidden_dim', None)
        model = MaxMViT_MLP_GMU(
            num_classes=num_classes, 
            hidden_size=hidden_size, 
            dropout_rate=dropout_rate,
            fusion_hidden_dim=fusion_hidden_dim
        )
        optimizers = get_optimizer_gmu(model, lr=lr)
        
    elif model_type == 'crossattn':
        num_heads = model_cfg.get('num_heads', 8)
        num_cross_layers = model_cfg.get('num_cross_layers', 2)
        fusion_type = model_cfg.get('fusion_type', 'concat')
        model = MaxMViT_MLP_CrossAttn(
            num_classes=num_classes,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            num_heads=num_heads,
            num_cross_layers=num_cross_layers,
            fusion_type=fusion_type
        )
        optimizers = get_optimizer_crossattn(model, lr=lr)

        
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
        
    return model, optimizers


def train_single_model(model_type, config, train_loader, val_loader, logger):
    """
    Train a single model variant and return results.
    
    Returns:
        dict with keys: model_type, best_val_acc, best_val_loss, best_epoch, total_time
    """
    train_cfg = config['training']
    model_cfg = config['model']
    
    DEVICE = torch.device(train_cfg.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    EPOCHS = train_cfg.get('epochs', 50)
    LR = train_cfg.get('lr', 0.0002)
    PATIENCE = train_cfg.get('patience', 10)
    
    # Get model
    num_classes = model_cfg.get('num_classes', 4)
    model, optimizers = get_model_and_optimizer(model_type, num_classes, LR, model_cfg)
    model.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"")
    logger.info(f"{'='*60}")
    logger.info(f"Training: {model_type.upper()}")
    logger.info(f"{'='*60}")
    logger.info(f"Total params: {total_params:,}")
    logger.info(f"Trainable params: {trainable_params:,}")
    
    # Schedulers
    sched_cfg = train_cfg.get('scheduler', {})
    schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max',  # Now monitoring accuracy (higher is better)
        factor=sched_cfg.get('factor', 0.1), 
        patience=sched_cfg.get('patience', 3), 
        min_lr=float(sched_cfg.get('min_lr', 1e-6))
    ) for opt in optimizers]
    
    criterion = nn.CrossEntropyLoss()
    
    # Training state
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        epoch_start = time.time()
        
        for batch_idx, (cqt, mel, label) in enumerate(train_loader):
            cqt, mel, label = cqt.to(DEVICE), mel.to(DEVICE), label.to(DEVICE)
            
            for opt in optimizers: opt.zero_grad()
            
            outputs = model(cqt, mel)
            loss = criterion(outputs, label)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            for opt in optimizers: opt.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation
        val_loss = 0
        val_correct = 0
        val_total = 0
        
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

        # Step Scheduler (monitoring accuracy now)
        for sch in schedulers: sch.step(val_acc)

        epoch_time = time.time() - epoch_start
        logger.info(f"[{model_type}] Epoch {epoch+1:02d} | Train [L:{train_loss:.4f} A:{train_acc:.1f}%] | Val [L:{val_loss:.4f} A:{val_acc:.1f}%] | Time: {epoch_time:.1f}s")
        
        # Early Stopping based on val accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            logger.info(f"[{model_type}] ★ New Best! Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"[{model_type}] Early stopping at epoch {epoch+1}")
                break
    
    total_time = time.time() - start_time
    
    result = {
        'model_type': model_type,
        'best_val_acc': best_val_acc,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'total_epochs': epoch + 1,
        'total_time': total_time,
        'total_params': total_params,
        'trainable_params': trainable_params
    }
    
    logger.info(f"[{model_type}] Finished! Best Acc: {best_val_acc:.2f}% at epoch {best_epoch}")
    
    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()
    
    return result


def generate_comparison_table(results, logger):
    """Generate a markdown comparison table from results."""
    
    # Prepare table data
    headers = ["Model", "Fusion Type", "Best Val Acc (%)", "Best Val Loss", "Best Epoch", "Total Epochs", "Time (min)", "Params (M)"]
    
    fusion_names = {
        'original': 'Concatenation',
        'gmu': 'Gated Multimodal Unit',
        'crossattn': 'Cross-Attention'
    }
    
    table_data = []
    for r in results:
        table_data.append([
            r['model_type'].upper(),
            fusion_names.get(r['model_type'], 'Unknown'),
            f"{r['best_val_acc']:.2f}",
            f"{r['best_val_loss']:.4f}",
            r['best_epoch'],
            r['total_epochs'],
            f"{r['total_time']/60:.1f}",
            f"{r['total_params']/1e6:.2f}"
        ])
    
    # Sort by best accuracy (descending)
    table_data.sort(key=lambda x: float(x[2]), reverse=True)
    
    # Generate table
    table_str = tabulate(table_data, headers=headers, tablefmt="pipe")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 80)
    logger.info("")
    logger.info(table_str)
    logger.info("")
    
    # Find winner
    winner = max(results, key=lambda x: x['best_val_acc'])
    logger.info(f"🏆 WINNER: {winner['model_type'].upper()} with {winner['best_val_acc']:.2f}% accuracy")
    
    return table_str


def train_compare(config_path):
    """Main function to train and compare all models."""
    
    # Load config
    config = load_config(config_path)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = config['dataset']['name']
    log_dir = config.get('paths', {}).get('log_dir', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"compare_{dataset_name}_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"="*80)
    logger.info(f"MODEL COMPARISON EXPERIMENT")
    logger.info(f"="*80)
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Timestamp: {timestamp}")
    
    # Seed
    SEED = config['training'].get('seed', 42)
    seed_everything(SEED)
    logger.info(f"Random seed: {SEED}")
    
    # Load data ONCE (shared across all models)
    logger.info("Loading dataset...")
    train_loader, val_loader = get_dataloaders(config)
    if not train_loader:
        logger.error("Failed to load data.")
        return
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Train all 3 models
    model_types = ['original', 'gmu', 'crossattn']
    results = []
    
    for model_type in model_types:
        seed_everything(SEED)  # Reset seed for fair comparison
        result = train_single_model(model_type, config, train_loader, val_loader, logger)
        results.append(result)
    
    # Generate comparison table
    table_str = generate_comparison_table(results, logger)
    
    # Save results as JSON
    results_file = os.path.join(log_dir, f"compare_{dataset_name}_{timestamp}_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {results_file}")
    
    # Save markdown table
    table_file = os.path.join(log_dir, f"compare_{dataset_name}_{timestamp}_table.md")
    with open(table_file, 'w') as f:
        f.write(f"# Model Comparison Results\n\n")
        f.write(f"**Dataset:** {dataset_name}\n\n")
        f.write(f"**Date:** {timestamp}\n\n")
        f.write(table_str)
        f.write("\n\n")
        
        # Add detailed results
        f.write("## Detailed Results\n\n")
        for r in results:
            f.write(f"### {r['model_type'].upper()}\n")
            f.write(f"- Best Val Accuracy: **{r['best_val_acc']:.2f}%**\n")
            f.write(f"- Best Val Loss: {r['best_val_loss']:.4f}\n")
            f.write(f"- Best Epoch: {r['best_epoch']}\n")
            f.write(f"- Total Epochs: {r['total_epochs']}\n")
            f.write(f"- Training Time: {r['total_time']/60:.1f} minutes\n")
            f.write(f"- Parameters: {r['total_params']:,}\n\n")
    
    logger.info(f"Markdown table saved to: {table_file}")
    logger.info("")
    logger.info("=" * 80)
    logger.info("EXPERIMENT COMPLETED!")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and compare all model variants")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()
    
    train_compare(args.config)
