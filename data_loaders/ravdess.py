import torch
import numpy as np
import librosa
import cv2
import os
import pickle
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Audio
from tqdm import tqdm

# Cache directory
CACHE_DIR = ".cache_spectrograms"


class RAVDESSHFDataset(Dataset):
    """Dataset class for RAVDESS from Hugging Face with caching."""
    
    def __init__(self, hf_id="TwinkStart/RAVDESS", split="ravdess_emo", 
                 target_classes=['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise'], 
                 sr=44100, target_size=(224, 224)):
        self.sr = sr
        self.target_size = target_size
        self.target_classes = target_classes
        self.class_map = {c: i for i, c in enumerate(target_classes)}
        
        # Normalization params (ImageNet)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        # Audio params
        self.n_fft = 4096
        self.hop_length = 256
        
        # Cache path
        os.makedirs(CACHE_DIR, exist_ok=True)
        safe_name = hf_id.replace("/", "_")
        self.cache_path = os.path.join(CACHE_DIR, f"{safe_name}_ravdess.pkl")
        
        # Load from cache or build
        if os.path.exists(self.cache_path):
            print(f"Loading from cache: {self.cache_path}")
            with open(self.cache_path, 'rb') as f:
                cache = pickle.load(f)
            self.data = cache['data']
            self.indices = cache['indices']
            print(f"Loaded {len(self.indices)} cached samples")
        else:
            print(f"Building cache (one-time operation)...")
            self._build_cache(hf_id, split)
    
    def _build_cache(self, hf_id, split):
        """Precompute all spectrograms and save to disk."""
        print(f"Loading {hf_id} [{split}]...")
        ds = load_dataset(hf_id, split=split).cast_column("audio", Audio(decode=False))
        
        # RAVDESS emotion mapping
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
        
        # Filter data
        self.indices = []
        for idx, item in enumerate(ds):
            raw_emo = item.get('emotion')
            if isinstance(raw_emo, str):
                raw_emo = raw_emo.lower().strip()
            short_emo = emo_map.get(raw_emo)
            if short_emo in self.target_classes:
                self.indices.append((idx, self.class_map[short_emo]))
        
        print(f"Precomputing {len(self.indices)} spectrograms...")
        
        import soundfile as sf
        import io
        
        self.data = {}
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
                
                # Compute spectrograms
                cqt = librosa.cqt(y, sr=self.sr)
                cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
                
                mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                
                # Process spectrograms
                cqt_img = self._resize_normalize(cqt_db)
                mel_img = self._resize_normalize(mel_db)
                
                self.data[ds_idx] = {
                    'cqt': cqt_img.astype(np.float16),
                    'mel': mel_img.astype(np.float16),
                    'label': label
                }
            except Exception as e:
                print(f"Error processing {ds_idx}: {e}")
                dummy = np.zeros((3, self.target_size[0], self.target_size[1]), dtype=np.float16)
                self.data[ds_idx] = {'cqt': dummy, 'mel': dummy, 'label': label}
        
        # Save cache
        print(f"Saving cache to {self.cache_path}...")
        with open(self.cache_path, 'wb') as f:
            pickle.dump({'data': self.data, 'indices': self.indices}, f)
        print(f"Cache saved! ({os.path.getsize(self.cache_path) / 1e6:.1f} MB)")

    def _resize_normalize(self, spec):
        """Resize and normalize spectrogram."""
        spec_min, spec_max = spec.min(), spec.max()
        spec_norm = (spec - spec_min) / (spec_max - spec_min + 1e-8)
        spec_resized = cv2.resize(spec_norm, (self.target_size[1], self.target_size[0]))
        
        # Stack 3 times for RGB
        spec_3ch = np.stack([spec_resized]*3, axis=0)
        
        # ImageNet normalization
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


def get_ravdess_dataloaders(hf_id="TwinkStart/RAVDESS", batch_size=16, num_workers=4):
    """Get train and validation dataloaders."""
    import random
    import copy
    
    full_ds = RAVDESSHFDataset(hf_id, split="ravdess_emo")
    
    # Split 80/20
    full_indices = full_ds.indices.copy()
    random.seed(42)
    random.shuffle(full_indices)
    
    total = len(full_indices)
    val_len = int(total * 0.2)
    train_len = total - val_len
    
    train_indices = full_indices[:train_len]
    val_indices = full_indices[train_len:]
    
    train_ds = copy.deepcopy(full_ds)
    train_ds.indices = train_indices
    
    val_ds = copy.deepcopy(full_ds)
    val_ds.indices = val_indices
    
    print(f"Split RAVDESS: Train {len(train_ds)}, Val {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader
