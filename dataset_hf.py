import torch
import numpy as np
import librosa
import cv2
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

class IEMOCAPHFDataset(Dataset):
    def __init__(self, hf_id="AbstractTTS/IEMOCAP", split="train", target_classes=['neu', 'hap', 'ang', 'sad'], sr=44100, target_size=(244, 244)):
        """
        Dataset class for IEMOCAP from Hugging Face.
        
        Args:
            hf_id (str): Hugging Face dataset ID.
            split (str): Dataset split ('train', 'validation', etc.).
            target_classes (list): List of emotions to classify.
            sr (int): Sampling rate.
            target_size (tuple): Spec image size.
        """
        self.sr = sr
        self.target_size = target_size
        self.target_classes = target_classes
        self.class_map = {c: i for i, c in enumerate(target_classes)}
        
        print(f"Loading {hf_id} [{split}]...")
        # Load dataset
        # Disable auto-decoding to avoid torchcodec issues
        from datasets import Audio
        self.ds = load_dataset(hf_id, split=split).cast_column("audio", Audio(decode=False))
        
        # Audio params
        self.n_fft = 4096
        self.hop_length = 256
        
        # Mapping from dataset emotion strings to our codes
        # IEMOCAP usually: neutral, happy, angry, sad, frustrated, excited, fear, surprise, disgust, other, xxx
        # 'exc' -> 'hap' is standard.
        self.filter_data()

    def filter_data(self):
        self.indices = []
        for idx, item in enumerate(self.ds):
            emo = item.get('major_emotion')
            
            # Map full names to abbreviations if needed or just use first 3 chars if consistent
            # Sample showed 'neutral'. Let's assume standard full names or use mapping
            # Standard IEMOCAP mapping often used:
            # ang: angry
            # hap: happy, excited
            # sad: sad
            # neu: neutral
            
            if emo == 'excited':
                emo = 'happy'
                
            # Convert to 3-char code for consistency with my other code if needed, 
            # Or just map the string 'happy' to what my target_classes expects.
            # My target_classes in train script are ['neu', 'hap', 'ang', 'sad'].
            # So I should map 'neutral' -> 'neu', 'happy' -> 'hap', 'angry' -> 'ang', 'sad' -> 'sad'.
            
            emo_map = {
                'neutral': 'neu',
                'happy': 'hap',
                'excited': 'hap',
                'angry': 'ang',
                'sad': 'sad'
            }
            
            short_emo = emo_map.get(emo)
            
            if short_emo in self.target_classes:
                self.indices.append((idx, self.class_map[short_emo]))
                
        print(f"Filtered {len(self.indices)} samples from {len(self.ds)} total.")

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        ds_idx, label = self.indices[idx]
        item = self.ds[ds_idx]
        
        # Audio processing
        # item['audio']['array'] is numpy array
        # item['audio']['sampling_rate'] is original SR
        # Audio processing
        # item['audio']['bytes'] contains raw audio bytes when decode=False
        audio_bytes = item['audio']['bytes']
        
        # Decode with soundfile
        import soundfile as sf
        import io
        y, orig_sr = sf.read(io.BytesIO(audio_bytes))
        
        # Resample if needed
        # Resample if needed
        if orig_sr != self.sr:
            # Resample needs float32
            y = y.astype(np.float32)
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=self.sr)
        else:
            y = y.astype(np.float32)
            
        # Ensure y is 1D
        if y.ndim > 1:
            y = np.mean(y, axis=0) # Convert stereo to mono
            
        # --- Preprocessing same as SERDataset ---
        
        # CQT
        try:
            cqt = librosa.cqt(y, sr=self.sr)
            cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
            
            # Mel-STFT
            mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            
            # Resize & Normalize
            cqt_img = self._resize_normalize(cqt_db)
            mel_img = self._resize_normalize(mel_db)
            
            cqt_tensor = torch.tensor(cqt_img, dtype=torch.float32).unsqueeze(0)
            mel_tensor = torch.tensor(mel_img, dtype=torch.float32).unsqueeze(0)
            
            return cqt_tensor, mel_tensor, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            # In case of empty audio or error, return correct shapes with zeros
            print(f"Error processing audio sample {ds_idx}: {e}")
            dummy_img = torch.zeros((1, self.target_size[0], self.target_size[1]), dtype=torch.float32)
            return dummy_img, dummy_img, torch.tensor(label, dtype=torch.long)

    def _resize_normalize(self, spec):
        spec_min = spec.min()
        spec_max = spec.max()
        spec_norm = (spec - spec_min) / (spec_max - spec_min + 1e-8)
        spec_resized = cv2.resize(spec_norm, (self.target_size[1], self.target_size[0]))
        return spec_resized

def get_hf_dataloaders(hf_id, batch_size=32, num_workers=4):
    # Hugging Face datasets usually have 'train', 'validation', 'test' splits or just 'train'.
    # AbstractTTS/IEMOCAP structure: checking...
    # If standard splits exist:
    try:
        train_ds = IEMOCAPHFDataset(hf_id, split="train")
        # Try to load validation/test if exists, else split train
        try:
            val_ds = IEMOCAPHFDataset(hf_id, split="validation")
        except:
             # If no validation split, manually split train using datasets library feature
             print("No validation split found. Automatically splitting train set (80/20)...")
             # We need to access the underlying HF dataset object to split it
             # BUT IEMOCAPHFDataset wraps it and applies filtering in __init__.
             # Splitting the filtered indices is cleaner.
             
             # Let's split indices of train_ds
             full_indices = train_ds.indices
             total_len = len(full_indices)
             val_len = int(total_len * 0.2)
             train_len = total_len - val_len
             
             # Random shuffle
             import random
             random.shuffle(full_indices)
             
             train_indices = full_indices[:train_len]
             val_indices = full_indices[train_len:]
             
             # Assign back to train_ds
             train_ds.indices = train_indices
             
             # Create val_ds as a copy but with val indices
             import copy
             val_ds = copy.deepcopy(train_ds) 
             val_ds.indices = val_indices
             print(f"Split complete. Train: {len(train_ds)}, Val: {len(val_ds)}")
             
        test_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers) if val_ds else None
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
        return train_loader, test_loader
    except Exception as e:
        print(f"Dataset load error: {e}")
        return None, None
