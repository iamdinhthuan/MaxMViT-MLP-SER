
import argparse
import torch
import torch.nn as nn
import time
import os
import pickle
from datetime import datetime
from utils import load_config, seed_everything
from model import MaxMViT_MLP, get_optimizer
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa
import cv2
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Audio
import copy
import random
from tqdm import tqdm

# ============================================================
# CACHED DATASET
# ============================================================

CACHE_DIR = ".cache_spectrograms"


def get_cache_path(hf_id):
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe_name = hf_id.replace("/", "_")
    return os.path.join(CACHE_DIR, f"{safe_name}_original.pkl")


class CachedDataset(Dataset):
    """Dataset with caching."""
    
    def __init__(self, hf_id="AbstractTTS/IEMOCAP", split="train", 
                 target_classes=['neu', 'hap', 'ang', 'sad'], sr=44100, target_size=(244, 244)):
        self.sr = sr
        self.target_size = target_size
        self.target_classes = target_classes
        self.class_map = {c: i for i, c in enumerate(target_classes)}
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.n_fft = 4096
        self.hop_length = 256
        
        cache_path = get_cache_path(hf_id)
        
        if os.path.exists(cache_path):
            print(f"Loading from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
            self.data = cache['data']
            self.indices = cache['indices']
            print(f"Loaded {len(self.indices)} cached samples")
        else:
            print(f"Building cache...")
            self._build_cache(hf_id, split, cache_path)
    
    def _build_cache(self, hf_id, split, cache_path):
        print(f"Loading {hf_id} [{split}]...")
        ds = load_dataset(hf_id, split=split).cast_column("audio", Audio(decode=False))
        
        emo_map = {'neutral': 'neu', 'happy': 'hap', 'angry': 'ang', 'sad': 'sad'}
        self.indices = []
        self.data = {}
        
        for idx, item in enumerate(ds):
            emo = item.get('major_emotion')
            short_emo = emo_map.get(emo)
            if short_emo in self.target_classes:
                self.indices.append((idx, self.class_map[short_emo]))
        
        print(f"Precomputing {len(self.indices)} spectrograms...")
        
        import soundfile as sf
        import io
        
        for i, (ds_idx, label) in enumerate(tqdm(self.indices, desc="Caching")):
            item = ds[ds_idx]
            audio_bytes = item['audio']['bytes']
            
            try:
                y, orig_sr = sf.read(io.BytesIO(audio_bytes))
                
                if orig_sr != self.sr:
                    y = y.astype(np.float32)
                    y = librosa.resample(y, orig_sr=orig_sr, target_sr=self.sr)
                else:
                    y = y.astype(np.float32)
                    
                if y.ndim > 1:
                    y = np.mean(y, axis=0)

                if len(y) < self.n_fft:
                    y = np.pad(y, (0, self.n_fft - len(y) + 1), mode='constant')
                
                cqt = librosa.cqt(y, sr=self.sr)
                cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
                
                mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                
                cqt_img = self._resize_normalize(cqt_db)
                mel_img = self._resize_normalize(mel_db)
                
                self.data[ds_idx] = {
                    'cqt': cqt_img.astype(np.float16),
                    'mel': mel_img.astype(np.float16),
                    'label': label
                }
            except Exception as e:
                dummy = np.zeros((3, self.target_size[0], self.target_size[1]), dtype=np.float16)
                self.data[ds_idx] = {'cqt': dummy, 'mel': dummy, 'label': label}
        
        print(f"Saving cache to {cache_path}...")
        with open(cache_path, 'wb') as f:
            pickle.dump({'data': self.data, 'indices': self.indices}, f)

    def _resize_normalize(self, spec):
        spec_min, spec_max = spec.min(), spec.max()
        spec_norm = (spec - spec_min) / (spec_max - spec_min + 1e-8)
        spec_resized = cv2.resize(spec_norm, (self.target_size[1], self.target_size[0]))
        spec_3ch = np.stack([spec_resized]*3, axis=0)
        for i in range(3):
            spec_3ch[i] = (spec_3ch[i] - self.mean[i]) / self.std[i]
        return spec_3ch

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        ds_idx, label = self.indices[idx]
        item = self.data[ds_idx]
        return (torch.tensor(item['cqt'].astype(np.float32), dtype=torch.float32),
                torch.tensor(item['mel'].astype(np.float32), dtype=torch.float32),
                torch.tensor(item['label'], dtype=torch.long))


def get_dataloaders(config):
    """Get train/val/test dataloaders with 80/10/10 split."""
    ds_config = config.get('dataset', {})
    hf_id = ds_config.get('args', {}).get('hf_id', 'AbstractTTS/IEMOCAP')
    batch_size = ds_config.get('args', {}).get('batch_size', 16)
    num_workers = ds_config.get('args', {}).get('num_workers', 8)
    
    full_ds = CachedDataset(hf_id, split="train")
    
    full_indices = full_ds.indices.copy()
    random.seed(42)
    random.shuffle(full_indices)
    
    n_total = len(full_indices)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    
    train_indices = full_indices[:n_train]
    val_indices = full_indices[n_train:n_train + n_val]
    test_indices = full_indices[n_train + n_val:]
    
    train_ds = copy.deepcopy(full_ds)
    train_ds.indices = train_indices
    
    val_ds = copy.deepcopy(full_ds)
    val_ds.indices = val_indices
    
    test_ds = copy.deepcopy(full_ds)
    test_ds.indices = test_indices
    
    print(f"Split: Train {len(train_ds)}, Val {len(val_ds)}, Test {len(test_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader


def train_with_lr(config, lr, log_file, train_loader, val_loader):
    """Train model with specific learning rate."""
    train_cfg = config['training']
    model_cfg = config['model']
    
    SEED = train_cfg.get('seed', 42)
    seed_everything(SEED)
    
    DEVICE = torch.device(train_cfg.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    EPOCHS = train_cfg.get('epochs', 50)
    PATIENCE = train_cfg.get('patience', 5)
    
    # Model
    num_classes = model_cfg.get('num_classes', 4)
    model = MaxMViT_MLP(num_classes=num_classes)
    model.to(DEVICE)
    
    # Optimization with specified LR
    optimizers = get_optimizer(model, lr=lr)
    sched_cfg = train_cfg.get('scheduler', {})
    schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', 
        factor=sched_cfg.get('factor', 0.1), 
        patience=sched_cfg.get('patience', 2), 
        min_lr=float(sched_cfg.get('min_lr', 1e-6))
    ) for opt in optimizers]
    
    criterion = nn.CrossEntropyLoss()
    
    lr_str = f"{lr:.0e}" if lr < 0.001 else f"{lr}"
    print(f"\n{'='*60}")
    print(f"Training with LR = {lr_str}")
    print(f"{'='*60}")
    
    best_val_acc = 0.0
    patience_counter = 0
    best_state = None
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        start_time = time.time()
        
        for batch_idx, (cqt, mel, label) in enumerate(train_loader):
            cqt, mel, label = cqt.to(DEVICE, non_blocking=True), mel.to(DEVICE, non_blocking=True), label.to(DEVICE, non_blocking=True)
            
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
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for cqt, mel, label in val_loader:
                cqt, mel, label = cqt.to(DEVICE, non_blocking=True), mel.to(DEVICE, non_blocking=True), label.to(DEVICE, non_blocking=True)
                outputs = model(cqt, mel)
                loss = criterion(outputs, label)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += label.size(0)
                val_correct += predicted.eq(label).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        for sch in schedulers: sch.step(val_loss)
        
        epoch_time = time.time() - start_time
        log_msg = f"[LR={lr_str}] Epoch {epoch+1:02d} | Train [L:{train_loss:.4f} A:{train_acc:.1f}%] | Val [L:{val_loss:.4f} A:{val_acc:.1f}%] | Time: {epoch_time:.1f}s"
        print(log_msg)
        
        with open(log_file, 'a') as f:
            f.write(log_msg + '\n')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
            print(f"  -> New Best! Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  -> Early stopping at epoch {epoch+1}")
                break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return model, best_val_acc


def evaluate_test(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for cqt, mel, label in test_loader:
            cqt, mel, label = cqt.to(device, non_blocking=True), mel.to(device, non_blocking=True), label.to(device, non_blocking=True)
            outputs = model(cqt, mel)
            _, predicted = outputs.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
    return 100. * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    config = load_config(args.config)
    DEVICE = torch.device(config['training'].get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    
    # Learning rates to test
    learning_rates = [
        0.0002,    # Original (baseline)
        0.0001,    # Lower
        0.00005,   # Even lower
        0.00002,   # Very low
    ]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"lr_compare_{config.get('experiment_name', 'exp')}_{timestamp}.log"
    
    print(f"\n{'#'*60}")
    print(f"# LEARNING RATE COMPARISON")
    print(f"# Config: {args.config}")
    print(f"# Log: {log_file}")
    print(f"# LRs to test: {learning_rates}")
    print(f"{'#'*60}\n")
    
    with open(log_file, 'w') as f:
        f.write(f"Learning Rate Comparison - {timestamp}\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"LRs: {learning_rates}\n")
        f.write("="*60 + "\n\n")
    
    # Load data once
    train_loader, val_loader, test_loader = get_dataloaders(config)
    
    results = {}
    
    for lr in learning_rates:
        lr_str = f"{lr:.0e}" if lr < 0.001 else f"{lr}"
        
        with open(log_file, 'a') as f:
            f.write(f"\n--- LR = {lr_str} ---\n")
        
        model, val_acc = train_with_lr(config, lr, log_file, train_loader, val_loader)
        test_acc = evaluate_test(model, test_loader, DEVICE)
        
        results[lr] = {'val': val_acc, 'test': test_acc}
        print(f"[LR={lr_str}] Test Accuracy: {test_acc:.2f}%\n")
        
        with open(log_file, 'a') as f:
            f.write(f"[LR={lr_str}] Test Accuracy: {test_acc:.2f}%\n")
    
    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS - Learning Rate Comparison")
    print("="*60)
    print(f"{'LR':<12} {'Val Acc':<12} {'Test Acc':<12}")
    print("-"*36)
    
    summary_lines = ["\n" + "="*60, "FINAL RESULTS", "="*60]
    
    best_test_acc = 0
    best_lr = None
    
    for lr, accs in results.items():
        lr_str = f"{lr:.0e}" if lr < 0.001 else f"{lr}"
        line = f"{lr_str:<12} {accs['val']:.2f}%       {accs['test']:.2f}%"
        print(line)
        summary_lines.append(line)
        
        if accs['test'] > best_test_acc:
            best_test_acc = accs['test']
            best_lr = lr
    
    best_lr_str = f"{best_lr:.0e}" if best_lr < 0.001 else f"{best_lr}"
    conclusion = f"\nBest LR: {best_lr_str} with Test Acc: {best_test_acc:.2f}%"
    print(conclusion)
    summary_lines.append(conclusion)
    
    with open(log_file, 'a') as f:
        f.write("\n".join(summary_lines))
    
    print(f"\nResults saved to: {log_file}")


if __name__ == "__main__":
    main()
