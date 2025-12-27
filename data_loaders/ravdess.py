
import torch
import numpy as np
import librosa
import cv2
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Audio
import logging

class RAVDESSHFDataset(Dataset):
    def __init__(self, hf_id="TwinkStart/RAVDESS", split="ravdess_emo", 
                 target_classes=['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise'], 
                 sr=44100, target_size=(224, 224), augment=False):
        """
        Dataset class for RAVDESS from Hugging Face.
        
        Args:
            augment (bool): Whether to apply SpecAugment (training only).
        """
        self.sr = sr
        self.target_size = target_size
        self.target_classes = target_classes
        self.class_map = {c: i for i, c in enumerate(target_classes)}
        self.augment = augment
        
        # Normalization params (ImageNet)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        print(f"Loading {hf_id} [{split}]... (augment={augment})")
        # Load dataset
        self.ds = load_dataset(hf_id, split=split).cast_column("audio", Audio(decode=False))
        
        # Audio params
        self.n_fft = 4096
        self.hop_length = 256
        
        self.filter_data()

    def filter_data(self):
        self.indices = []
        # RAVDESS usually has: neutral, calm, happy, sad, angry, fearful, disgust, surprised
        # Mapping to 8 classes
        emo_map = {
            'neutral': 'neutral',
            'calm': 'calm',
            'happy': 'happy',
            'sad': 'sad',
            'angry': 'angry',
            'fearful': 'fear',
            'disgust': 'disgust',
            'surprised': 'surprise'
        }
        
        for idx, item in enumerate(self.ds):
            raw_emo = item.get('emotion') # e.g., "angry"
            
            # Simple normalization just in case
            if isinstance(raw_emo, str):
                raw_emo = raw_emo.lower().strip()
                
            short_emo = emo_map.get(raw_emo)
            
            if short_emo in self.target_classes:
                self.indices.append((idx, self.class_map[short_emo]))
                
        print(f"Filtered {len(self.indices)} samples from {len(self.ds)} total.")

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        ds_idx, label = self.indices[idx]
        item = self.ds[ds_idx]
        
        # Audio processing
        audio_bytes = item['audio']['bytes']
        
        # Decode
        import soundfile as sf
        import io
        y, orig_sr = sf.read(io.BytesIO(audio_bytes))
        
        # Resample
        if orig_sr != self.sr:
            y = y.astype(np.float32)
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=self.sr)
        else:
            y = y.astype(np.float32)
            
        # Ensure mono
        if y.ndim > 1:
            y = np.mean(y, axis=0) 
            
        # Fix: Pad short audio to prevent Librosa warnings
        if len(y) < self.n_fft:
            padding = self.n_fft - len(y) + 1
            y = np.pad(y, (0, padding), mode='constant')
            
        # --- Preprocessing ---
        try:
            # CQT
            cqt = librosa.cqt(y, sr=self.sr)
            cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
            
            # Mel-STFT
            mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            
            # Apply SpecAugment if training
            if self.augment:
                cqt_db = self._spec_augment(cqt_db)
                mel_db = self._spec_augment(mel_db)
            
            # Resize & Normalize with Delta Features
            cqt_img = self._resize_normalize_delta(cqt_db)
            mel_img = self._resize_normalize_delta(mel_db)
            
            cqt_tensor = torch.tensor(cqt_img, dtype=torch.float32)
            mel_tensor = torch.tensor(mel_img, dtype=torch.float32)
            
            return cqt_tensor, mel_tensor, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error processing sample {ds_idx}: {e}")
            dummy_img = torch.zeros((3, self.target_size[0], self.target_size[1]), dtype=torch.float32)
            return dummy_img, dummy_img, torch.tensor(label, dtype=torch.long)

    def _spec_augment(self, spec, time_mask_param=30, freq_mask_param=15, num_masks=2):
        """
        Apply SpecAugment: time masking and frequency masking.
        """
        spec = spec.copy()
        freq_bins, time_steps = spec.shape
        
        # Time masking
        for _ in range(num_masks):
            if time_steps > time_mask_param:
                t = np.random.randint(1, time_mask_param)
                t0 = np.random.randint(0, time_steps - t)
                spec[:, t0:t0+t] = spec.min()
        
        # Frequency masking
        for _ in range(num_masks):
            if freq_bins > freq_mask_param:
                f = np.random.randint(1, freq_mask_param)
                f0 = np.random.randint(0, freq_bins - f)
                spec[f0:f0+f, :] = spec.min()
        
        return spec
    
    def _normalize_single(self, spec):
        """Normalize a single spectrogram to [0, 1]."""
        spec_min = spec.min()
        spec_max = spec.max()
        return (spec - spec_min) / (spec_max - spec_min + 1e-8)
    
    def _resize_normalize_delta(self, spec):
        """
        Create 3-channel image with Delta features:
        - Channel 0: Original spectrogram
        - Channel 1: Delta (velocity)
        - Channel 2: Delta-Delta (acceleration)
        """
        delta = librosa.feature.delta(spec, order=1)
        delta2 = librosa.feature.delta(spec, order=2)
        
        # Normalize each independently
        spec_norm = self._normalize_single(spec)
        delta_norm = self._normalize_single(delta)
        delta2_norm = self._normalize_single(delta2)
        
        # Resize
        spec_resized = cv2.resize(spec_norm, (self.target_size[1], self.target_size[0]))
        delta_resized = cv2.resize(delta_norm, (self.target_size[1], self.target_size[0]))
        delta2_resized = cv2.resize(delta2_norm, (self.target_size[1], self.target_size[0]))
        
        # Stack as 3 channels
        spec_3ch = np.stack([spec_resized, delta_resized, delta2_resized], axis=0)
        
        # ImageNet Norm
        for i in range(3):
            spec_3ch[i] = (spec_3ch[i] - self.mean[i]) / self.std[i]
            
        return spec_3ch

    def _resize_normalize(self, spec):
        """Legacy method."""
        return self._resize_normalize_delta(spec)

def get_ravdess_dataloaders(hf_id="TwinkStart/RAVDESS", batch_size=16, num_workers=4):
    try:
        full_ds = RAVDESSHFDataset(hf_id, split="ravdess_emo", augment=True)
        
        # Manual Split 80/20
        full_indices = full_ds.indices
        total = len(full_indices)
        val_len = int(total * 0.2)
        train_len = total - val_len
        
        import random
        # Ensure reproducibility
        random.seed(42) 
        random.shuffle(full_indices)
        
        train_indices = full_indices[:train_len]
        val_indices = full_indices[train_len:]
        
        import copy
        train_ds = copy.deepcopy(full_ds)
        train_ds.indices = train_indices
        train_ds.augment = True  # Training with augmentation
        
        val_ds = copy.deepcopy(full_ds)
        val_ds.indices = val_indices
        val_ds.augment = False  # Validation without augmentation
        
        print(f"Split RAVDESS: Train {len(train_ds)} (augment=True), Val {len(val_ds)} (augment=False)")
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        test_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        return train_loader, test_loader
    except Exception as e:
        print(f"RAVDESS load error: {e}")
        return None, None
