
import argparse
import torch
import torch.nn as nn
import time
import os
import logging
import sys
import pickle
import hashlib
from datetime import datetime
from utils import load_config, setup_logging, seed_everything
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
# CACHED DATASET CLASSES - Precompute spectrograms once
# ============================================================

CACHE_DIR = ".cache_spectrograms"


def get_cache_path(hf_id, mode):
    """Generate cache file path based on dataset and mode."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe_name = hf_id.replace("/", "_")
    return os.path.join(CACHE_DIR, f"{safe_name}_{mode}.pkl")


class CachedDatasetOriginal(Dataset):
    """Original dataset with CACHING - stack same spectrogram 3 times."""
    
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
        
        cache_path = get_cache_path(hf_id, "original")
        
        if os.path.exists(cache_path):
            print(f"[ORIGINAL] Loading from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
            self.data = cache['data']
            self.indices = cache['indices']
            print(f"[ORIGINAL] Loaded {len(self.indices)} cached samples")
        else:
            print(f"[ORIGINAL] Building cache (one-time operation)...")
            self._build_cache(hf_id, split, cache_path)
    
    def _build_cache(self, hf_id, split, cache_path):
        """Precompute all spectrograms and save to disk."""
        print(f"[ORIGINAL] Loading {hf_id} [{split}]...")
        ds = load_dataset(hf_id, split=split).cast_column("audio", Audio(decode=False))
        
        # Filter data
        emo_map = {'neutral': 'neu', 'happy': 'hap', 'angry': 'ang', 'sad': 'sad'}
        self.indices = []
        self.data = {}
        
        for idx, item in enumerate(ds):
            emo = item.get('major_emotion')
            short_emo = emo_map.get(emo)
            if short_emo in self.target_classes:
                self.indices.append((idx, self.class_map[short_emo]))
        
        print(f"[ORIGINAL] Precomputing {len(self.indices)} spectrograms...")
        
        import soundfile as sf
        import io
        
        for i, (ds_idx, label) in enumerate(tqdm(self.indices, desc="[ORIGINAL] Caching")):
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
                
                # Compute spectrograms
                cqt = librosa.cqt(y, sr=self.sr)
                cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
                
                mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                
                # Process and store
                cqt_img = self._resize_normalize_original(cqt_db)
                mel_img = self._resize_normalize_original(mel_db)
                
                self.data[ds_idx] = {
                    'cqt': cqt_img.astype(np.float16),  # Use float16 to save memory
                    'mel': mel_img.astype(np.float16),
                    'label': label
                }
            except Exception as e:
                print(f"Error processing {ds_idx}: {e}")
                dummy = np.zeros((3, self.target_size[0], self.target_size[1]), dtype=np.float16)
                self.data[ds_idx] = {'cqt': dummy, 'mel': dummy, 'label': label}
        
        # Save cache
        print(f"[ORIGINAL] Saving cache to {cache_path}...")
        with open(cache_path, 'wb') as f:
            pickle.dump({'data': self.data, 'indices': self.indices}, f)
        print(f"[ORIGINAL] Cache saved! ({os.path.getsize(cache_path) / 1e6:.1f} MB)")

    def _resize_normalize_original(self, spec):
        """Original method: stack same spectrogram 3 times."""
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
        return (torch.tensor(item['cqt'], dtype=torch.float32),
                torch.tensor(item['mel'], dtype=torch.float32),
                torch.tensor(item['label'], dtype=torch.long))


class CachedDatasetImproved(Dataset):
    """Improved dataset with CACHING - delta features + SpecAugment."""
    
    def __init__(self, hf_id="AbstractTTS/IEMOCAP", split="train", 
                 target_classes=['neu', 'hap', 'ang', 'sad'], sr=44100, target_size=(244, 244), augment=False):
        self.sr = sr
        self.target_size = target_size
        self.target_classes = target_classes
        self.class_map = {c: i for i, c in enumerate(target_classes)}
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.n_fft = 4096
        self.hop_length = 256
        self.augment = augment
        
        cache_path = get_cache_path(hf_id, "improved")
        
        if os.path.exists(cache_path):
            print(f"[IMPROVED] Loading from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
            self.data = cache['data']
            self.indices = cache['indices']
            print(f"[IMPROVED] Loaded {len(self.indices)} cached samples (augment={augment})")
        else:
            print(f"[IMPROVED] Building cache (one-time operation)...")
            self._build_cache(hf_id, split, cache_path)
    
    def _build_cache(self, hf_id, split, cache_path):
        """Precompute all spectrograms with delta features."""
        print(f"[IMPROVED] Loading {hf_id} [{split}]...")
        ds = load_dataset(hf_id, split=split).cast_column("audio", Audio(decode=False))
        
        emo_map = {'neutral': 'neu', 'happy': 'hap', 'angry': 'ang', 'sad': 'sad'}
        self.indices = []
        self.data = {}
        
        for idx, item in enumerate(ds):
            emo = item.get('major_emotion')
            short_emo = emo_map.get(emo)
            if short_emo in self.target_classes:
                self.indices.append((idx, self.class_map[short_emo]))
        
        print(f"[IMPROVED] Precomputing {len(self.indices)} spectrograms with delta features...")
        
        import soundfile as sf
        import io
        
        for i, (ds_idx, label) in enumerate(tqdm(self.indices, desc="[IMPROVED] Caching")):
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
                
                # Compute spectrograms
                cqt = librosa.cqt(y, sr=self.sr)
                cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
                
                mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                
                # Process with delta features (NO augment during caching)
                cqt_img = self._resize_normalize_delta(cqt_db)
                mel_img = self._resize_normalize_delta(mel_db)
                
                self.data[ds_idx] = {
                    'cqt': cqt_img.astype(np.float16),
                    'mel': mel_img.astype(np.float16),
                    'label': label
                }
            except Exception as e:
                print(f"Error processing {ds_idx}: {e}")
                dummy = np.zeros((3, self.target_size[0], self.target_size[1]), dtype=np.float16)
                self.data[ds_idx] = {'cqt': dummy, 'mel': dummy, 'label': label}
        
        print(f"[IMPROVED] Saving cache to {cache_path}...")
        with open(cache_path, 'wb') as f:
            pickle.dump({'data': self.data, 'indices': self.indices}, f)
        print(f"[IMPROVED] Cache saved! ({os.path.getsize(cache_path) / 1e6:.1f} MB)")

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
        
        spec_3ch = np.stack([spec_resized, delta_resized, delta2_resized], axis=0)
        for i in range(3):
            spec_3ch[i] = (spec_3ch[i] - self.mean[i]) / self.std[i]
        return spec_3ch
    
    def _spec_augment(self, spec, time_mask_param=30, freq_mask_param=15, num_masks=2):
        """Apply SpecAugment on-the-fly during training."""
        spec = spec.copy()
        _, h, w = spec.shape  # [3, H, W]
        for _ in range(num_masks):
            if w > time_mask_param:
                t = np.random.randint(1, time_mask_param)
                t0 = np.random.randint(0, w - t)
                spec[:, :, t0:t0+t] = 0  # Mask across all channels
        for _ in range(num_masks):
            if h > freq_mask_param:
                f = np.random.randint(1, freq_mask_param)
                f0 = np.random.randint(0, h - f)
                spec[:, f0:f0+f, :] = 0
        return spec

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        ds_idx, label = self.indices[idx]
        item = self.data[ds_idx]
        
        cqt = item['cqt'].astype(np.float32)
        mel = item['mel'].astype(np.float32)
        
        # Apply SpecAugment on-the-fly if training
        if self.augment:
            cqt = self._spec_augment(cqt)
            mel = self._spec_augment(mel)
        
        return (torch.tensor(cqt, dtype=torch.float32),
                torch.tensor(mel, dtype=torch.float32),
                torch.tensor(item['label'], dtype=torch.long))


# ============================================================
# DATALOADERS AND TRAINING
# ============================================================

def get_dataloaders(config, mode="original"):
    """Get dataloaders with caching.
    
    Args:
        mode: 'original', 'delta_only', or 'delta_augment'
    """
    ds_config = config.get('dataset', {})
    hf_id = ds_config.get('args', {}).get('hf_id', 'AbstractTTS/IEMOCAP')
    batch_size = ds_config.get('args', {}).get('batch_size', 16)
    num_workers = ds_config.get('args', {}).get('num_workers', 8)
    
    if mode == "original":
        train_ds = CachedDatasetOriginal(hf_id, split="train")
    else:
        # Both delta_only and delta_augment use CachedDatasetImproved
        # But delta_only has augment=False
        use_augment = (mode == "delta_augment")
        train_ds = CachedDatasetImproved(hf_id, split="train", augment=use_augment)
    
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
    if hasattr(val_ds, 'augment'):
        val_ds.augment = False  # Never augment validation
    
    print(f"Split: Train {len(train_ds)}, Val {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader


def train_single(config, mode_name, mode, log_file):
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
    train_loader, val_loader = get_dataloaders(config, mode=mode)
    
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
    parser.add_argument("--clear-cache", action="store_true", help="Clear cached spectrograms")
    args = parser.parse_args()
    
    # Clear cache if requested
    if args.clear_cache and os.path.exists(CACHE_DIR):
        import shutil
        shutil.rmtree(CACHE_DIR)
        print(f"Cleared cache directory: {CACHE_DIR}")
    
    config = load_config(args.config)
    
    # Create log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"compare_{config.get('experiment_name', 'exp')}_{timestamp}.log"
    
    print(f"\n{'#'*60}")
    print(f"# COMPARISON: ORIGINAL vs DELTA_ONLY vs DELTA+AUGMENT")
    print(f"# Config: {args.config}")
    print(f"# Log: {log_file}")
    print(f"# Cache: {CACHE_DIR}/")
    print(f"{'#'*60}\n")
    
    with open(log_file, 'w') as f:
        f.write(f"Comparison Training - {timestamp}\n")
        f.write(f"Config: {args.config}\n")
        f.write("="*60 + "\n\n")
    
    results = {}
    
    # PHASE 1: ORIGINAL (no delta, no augment)
    print("\n" + "="*60)
    print("PHASE 1: ORIGINAL (stack same spectrogram 3x, no augment)")
    print("="*60)
    
    with open(log_file, 'a') as f:
        f.write("PHASE 1: ORIGINAL\n")
    
    results['original'], _ = train_single(config, "ORIGINAL", mode="original", log_file=log_file)
    
    # PHASE 2: DELTA_ONLY (delta features, NO SpecAugment)
    print("\n" + "="*60)
    print("PHASE 2: DELTA_ONLY (delta features, NO SpecAugment)")
    print("="*60)
    
    with open(log_file, 'a') as f:
        f.write("\nPHASE 2: DELTA_ONLY\n")
    
    results['delta_only'], _ = train_single(config, "DELTA_ONLY", mode="delta_only", log_file=log_file)
    
    # PHASE 3: DELTA+AUGMENT (delta features + SpecAugment)
    print("\n" + "="*60)
    print("PHASE 3: DELTA+AUGMENT (delta features + SpecAugment)")
    print("="*60)
    
    with open(log_file, 'a') as f:
        f.write("\nPHASE 3: DELTA+AUGMENT\n")
    
    results['delta_augment'], _ = train_single(config, "DELTA+AUGMENT", mode="delta_augment", log_file=log_file)
    
    # Summary
    summary = f"""
{'='*60}
FINAL RESULTS
{'='*60}
ORIGINAL (stack 3x same)      : {results['original']:.2f}%
DELTA_ONLY (delta, no augment): {results['delta_only']:.2f}%
DELTA+AUGMENT (delta + augment): {results['delta_augment']:.2f}%

Delta vs Original: {results['delta_only'] - results['original']:+.2f}%
Augment Effect:    {results['delta_augment'] - results['delta_only']:+.2f}%
{'='*60}
"""
    print(summary)
    
    with open(log_file, 'a') as f:
        f.write(summary)
    
    print(f"\nResults saved to: {log_file}")


if __name__ == "__main__":
    main()
