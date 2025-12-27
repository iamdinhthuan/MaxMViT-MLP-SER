
import argparse
import torch
import torch.nn as nn
import time
import os
import logging
import sys
from datetime import datetime
from utils import load_config, setup_logging, seed_everything
from model import MaxMViT_MLP, get_optimizer
import warnings
warnings.filterwarnings("ignore")

# Import both original and improved data loaders
import numpy as np
import librosa
import cv2
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Audio
import copy
import random


class IEMOCAPDatasetOriginal(Dataset):
    """Original dataset WITHOUT delta features - stacks same spectrogram 3 times."""
    
    def __init__(self, hf_id="AbstractTTS/IEMOCAP", split="train", 
                 target_classes=['neu', 'hap', 'ang', 'sad'], sr=44100, target_size=(244, 244)):
        self.sr = sr
        self.target_size = target_size
        self.target_classes = target_classes
        self.class_map = {c: i for i, c in enumerate(target_classes)}
        
        print(f"[ORIGINAL] Loading {hf_id} [{split}]...")
        self.ds = load_dataset(hf_id, split=split).cast_column("audio", Audio(decode=False))
        
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.n_fft = 4096
        self.hop_length = 256
        
        self.filter_data()

    def filter_data(self):
        self.indices = []
        emo_map = {'neutral': 'neu', 'happy': 'hap', 'angry': 'ang', 'sad': 'sad'}
        for idx, item in enumerate(self.ds):
            emo = item.get('major_emotion')
            short_emo = emo_map.get(emo)
            if short_emo in self.target_classes:
                self.indices.append((idx, self.class_map[short_emo]))
        print(f"[ORIGINAL] Filtered {len(self.indices)} samples")

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        ds_idx, label = self.indices[idx]
        item = self.ds[ds_idx]
        
        import soundfile as sf
        import io
        audio_bytes = item['audio']['bytes']
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
            
        try:
            cqt = librosa.cqt(y, sr=self.sr)
            cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
            
            mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            
            # ORIGINAL: Stack same spectrogram 3 times
            cqt_img = self._resize_normalize_original(cqt_db)
            mel_img = self._resize_normalize_original(mel_db)
            
            return torch.tensor(cqt_img, dtype=torch.float32), \
                   torch.tensor(mel_img, dtype=torch.float32), \
                   torch.tensor(label, dtype=torch.long)
        except Exception as e:
            dummy_img = torch.zeros((3, self.target_size[0], self.target_size[1]), dtype=torch.float32)
            return dummy_img, dummy_img, torch.tensor(label, dtype=torch.long)

    def _resize_normalize_original(self, spec):
        """Original method: stack same spectrogram 3 times."""
        spec_min, spec_max = spec.min(), spec.max()
        spec_norm = (spec - spec_min) / (spec_max - spec_min + 1e-8)
        spec_resized = cv2.resize(spec_norm, (self.target_size[1], self.target_size[0]))
        
        # Stack SAME image 3 times (original approach)
        spec_3ch = np.stack([spec_resized]*3, axis=0)
        
        for i in range(3):
            spec_3ch[i] = (spec_3ch[i] - self.mean[i]) / self.std[i]
        return spec_3ch


class IEMOCAPDatasetImproved(Dataset):
    """Improved dataset WITH delta features and SpecAugment."""
    
    def __init__(self, hf_id="AbstractTTS/IEMOCAP", split="train", 
                 target_classes=['neu', 'hap', 'ang', 'sad'], sr=44100, target_size=(244, 244), augment=False):
        self.sr = sr
        self.target_size = target_size
        self.target_classes = target_classes
        self.class_map = {c: i for i, c in enumerate(target_classes)}
        self.augment = augment
        
        print(f"[IMPROVED] Loading {hf_id} [{split}]... (augment={augment})")
        self.ds = load_dataset(hf_id, split=split).cast_column("audio", Audio(decode=False))
        
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.n_fft = 4096
        self.hop_length = 256
        
        self.filter_data()

    def filter_data(self):
        self.indices = []
        emo_map = {'neutral': 'neu', 'happy': 'hap', 'angry': 'ang', 'sad': 'sad'}
        for idx, item in enumerate(self.ds):
            emo = item.get('major_emotion')
            short_emo = emo_map.get(emo)
            if short_emo in self.target_classes:
                self.indices.append((idx, self.class_map[short_emo]))
        print(f"[IMPROVED] Filtered {len(self.indices)} samples")

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        ds_idx, label = self.indices[idx]
        item = self.ds[ds_idx]
        
        import soundfile as sf
        import io
        audio_bytes = item['audio']['bytes']
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
            
        try:
            cqt = librosa.cqt(y, sr=self.sr)
            cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
            
            mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            
            # Apply SpecAugment if training
            if self.augment:
                cqt_db = self._spec_augment(cqt_db)
                mel_db = self._spec_augment(mel_db)
            
            # IMPROVED: Use delta features
            cqt_img = self._resize_normalize_delta(cqt_db)
            mel_img = self._resize_normalize_delta(mel_db)
            
            return torch.tensor(cqt_img, dtype=torch.float32), \
                   torch.tensor(mel_img, dtype=torch.float32), \
                   torch.tensor(label, dtype=torch.long)
        except Exception as e:
            dummy_img = torch.zeros((3, self.target_size[0], self.target_size[1]), dtype=torch.float32)
            return dummy_img, dummy_img, torch.tensor(label, dtype=torch.long)

    def _spec_augment(self, spec, time_mask_param=30, freq_mask_param=15, num_masks=2):
        spec = spec.copy()
        freq_bins, time_steps = spec.shape
        for _ in range(num_masks):
            if time_steps > time_mask_param:
                t = np.random.randint(1, time_mask_param)
                t0 = np.random.randint(0, time_steps - t)
                spec[:, t0:t0+t] = spec.min()
        for _ in range(num_masks):
            if freq_bins > freq_mask_param:
                f = np.random.randint(1, freq_mask_param)
                f0 = np.random.randint(0, freq_bins - f)
                spec[f0:f0+f, :] = spec.min()
        return spec
    
    def _normalize_single(self, spec):
        spec_min, spec_max = spec.min(), spec.max()
        return (spec - spec_min) / (spec_max - spec_min + 1e-8)
    
    def _resize_normalize_delta(self, spec):
        """Improved method: Original + Delta + Delta-Delta."""
        delta = librosa.feature.delta(spec, order=1)
        delta2 = librosa.feature.delta(spec, order=2)
        
        spec_norm = self._normalize_single(spec)
        delta_norm = self._normalize_single(delta)
        delta2_norm = self._normalize_single(delta2)
        
        spec_resized = cv2.resize(spec_norm, (self.target_size[1], self.target_size[0]))
        delta_resized = cv2.resize(delta_norm, (self.target_size[1], self.target_size[0]))
        delta2_resized = cv2.resize(delta2_norm, (self.target_size[1], self.target_size[0]))
        
        # Stack as: [Original, Delta, Delta-Delta]
        spec_3ch = np.stack([spec_resized, delta_resized, delta2_resized], axis=0)
        
        for i in range(3):
            spec_3ch[i] = (spec_3ch[i] - self.mean[i]) / self.std[i]
        return spec_3ch


def get_dataloaders(config, use_improved=False):
    """Get dataloaders - original or improved based on flag."""
    ds_config = config.get('dataset', {})
    hf_id = ds_config.get('args', {}).get('hf_id', 'AbstractTTS/IEMOCAP')
    batch_size = ds_config.get('args', {}).get('batch_size', 16)
    num_workers = ds_config.get('args', {}).get('num_workers', 4)
    
    if use_improved:
        train_ds = IEMOCAPDatasetImproved(hf_id, split="train", augment=True)
    else:
        train_ds = IEMOCAPDatasetOriginal(hf_id, split="train")
    
    # Split 80/20
    full_indices = train_ds.indices.copy()
    random.seed(42)
    random.shuffle(full_indices)
    
    train_len = int(len(full_indices) * 0.8)
    train_indices = full_indices[:train_len]
    val_indices = full_indices[train_len:]
    
    train_ds.indices = train_indices
    
    val_ds = copy.deepcopy(train_ds)
    val_ds.indices = val_indices
    if use_improved:
        val_ds.augment = False
    
    print(f"Split: Train {len(train_ds)}, Val {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader


def train_single(config, mode_name, use_improved, log_file):
    """Train a single model with specified mode."""
    train_cfg = config['training']
    model_cfg = config['model']
    
    SEED = train_cfg.get('seed', 42)
    seed_everything(SEED)
    
    DEVICE = torch.device(train_cfg.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    EPOCHS = train_cfg.get('epochs', 50)
    LR = train_cfg.get('lr', 0.0002)
    PATIENCE = train_cfg.get('patience', 5)
    
    # Data
    train_loader, val_loader = get_dataloaders(config, use_improved=use_improved)
    
    # Model
    num_classes = model_cfg.get('num_classes', 4)
    model = MaxMViT_MLP(num_classes=num_classes)
    model.to(DEVICE)
    
    # Optimization
    optimizers = get_optimizer(model, lr=LR)
    sched_cfg = train_cfg.get('scheduler', {})
    schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', 
        factor=sched_cfg.get('factor', 0.1), 
        patience=sched_cfg.get('patience', 2), 
        min_lr=float(sched_cfg.get('min_lr', 1e-6))
    ) for opt in optimizers]
    
    criterion = nn.CrossEntropyLoss()
    
    # Training Loop
    print(f"\n{'='*60}")
    print(f"Training [{mode_name}]")
    print(f"{'='*60}")
    
    best_val_acc = 0.0
    patience_counter = 0
    results = []
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        start_time = time.time()
        
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
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
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
        
        for sch in schedulers: sch.step(val_loss)
        
        epoch_time = time.time() - start_time
        log_msg = f"[{mode_name}] Epoch {epoch+1:02d} | Train [L:{train_loss:.4f} A:{train_acc:.1f}%] | Val [L:{val_loss:.4f} A:{val_acc:.1f}%] | Time: {epoch_time:.1f}s"
        print(log_msg)
        
        with open(log_file, 'a') as f:
            f.write(log_msg + '\n')
        
        results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
        
        # Early Stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            print(f"  -> New Best! Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  -> Early stopping at epoch {epoch+1}")
                break
    
    return best_val_acc, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Create log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"compare_{config.get('experiment_name', 'exp')}_{timestamp}.log"
    
    print(f"\n{'#'*60}")
    print(f"# COMPARISON TRAINING: ORIGINAL vs IMPROVED")
    print(f"# Config: {args.config}")
    print(f"# Log: {log_file}")
    print(f"{'#'*60}\n")
    
    with open(log_file, 'w') as f:
        f.write(f"Comparison Training - {timestamp}\n")
        f.write(f"Config: {args.config}\n")
        f.write("="*60 + "\n\n")
    
    # Train ORIGINAL (no delta, no augment)
    print("\n" + "="*60)
    print("PHASE 1: ORIGINAL (stack same spectrogram 3x, no augment)")
    print("="*60)
    
    with open(log_file, 'a') as f:
        f.write("PHASE 1: ORIGINAL\n")
    
    original_best_acc, original_results = train_single(config, "ORIGINAL", use_improved=False, log_file=log_file)
    
    # Train IMPROVED (delta features + SpecAugment)
    print("\n" + "="*60)
    print("PHASE 2: IMPROVED (delta features + SpecAugment)")
    print("="*60)
    
    with open(log_file, 'a') as f:
        f.write("\nPHASE 2: IMPROVED\n")
    
    improved_best_acc, improved_results = train_single(config, "IMPROVED", use_improved=True, log_file=log_file)
    
    # Summary
    summary = f"""
{'='*60}
FINAL RESULTS
{'='*60}
ORIGINAL Best Val Accuracy: {original_best_acc:.2f}%
IMPROVED Best Val Accuracy: {improved_best_acc:.2f}%

Improvement: {improved_best_acc - original_best_acc:+.2f}%
{'='*60}
"""
    print(summary)
    
    with open(log_file, 'a') as f:
        f.write(summary)
    
    print(f"\nResults saved to: {log_file}")


if __name__ == "__main__":
    main()
