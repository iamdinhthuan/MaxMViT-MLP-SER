import os
import torch
import numpy as np
import librosa
import cv2
from torch.utils.data import Dataset, DataLoader

class SERDataset(Dataset):
    def __init__(self, audio_paths, labels, sr=44100, target_size=(244, 244)): # Paper sr=44.1kHz, size=244x244
        """
        Dataset class for Speech Emotion Recognition.
        
        Args:
            audio_paths (list): List of paths to audio files.
            labels (list): List of integer labels.
            sr (int): Sampling rate.
            target_size (tuple): Target spectrogram size (H, W).
        """
        self.audio_paths = audio_paths
        self.labels = labels
        self.sr = sr
        self.target_size = target_size
        
        # Mel-STFT parameters (Paper Section III.C)
        self.n_fft = 4096
        self.hop_length = 256
        
        # CQT parameters (Default/Standard)
        # Paper mentions "logarithmic frequency binning".
        
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        path = self.audio_paths[idx]
        label = self.labels[idx]
        
        # Load audio
        y, sr = librosa.load(path, sr=self.sr)
        
        # --- Generate CQT ---
        # "Constant-Q resolution... higher resolution at lower frequencies"
        cqt = librosa.cqt(y, sr=sr)
        cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
        
        # --- Generate Mel-STFT ---
        # "Frame length 4096 samples and a hop size of 256 samples"
        # "Logarithm of the energy values"
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # --- Resize & Normalize ---
        cqt_img = self._resize_normalize(cqt_db)
        mel_img = self._resize_normalize(mel_db)
        
        # To Tensor [1, H, W]
        cqt_tensor = torch.tensor(cqt_img, dtype=torch.float32).unsqueeze(0)
        mel_tensor = torch.tensor(mel_img, dtype=torch.float32).unsqueeze(0)
        
        return cqt_tensor, mel_tensor, torch.tensor(label, dtype=torch.long)
        
    def _resize_normalize(self, spec):
        # Normalize to 0-255 or 0-1. Vision models usually like 0-1 or standard normalization.
        # Paper doesn't specify normalization, but implicitly required for images.
        # Let's normalize globally to 0-1 per image.
        spec_min = spec.min()
        spec_max = spec.max()
        spec_norm = (spec - spec_min) / (spec_max - spec_min + 1e-8)
        
        # Resize
        # cv2.resize expects (W, H)
        spec_resized = cv2.resize(spec_norm, (self.target_size[1], self.target_size[0]))
        
        return spec_resized

def get_dataloader(paths, labels, batch_size=32, shuffle=True):
    dataset = SERDataset(paths, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
