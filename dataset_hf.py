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
        self.ds = load_dataset(hf_id, split=split)
        
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
        audio_array = item['audio']['array']
        orig_sr = item['audio']['sampling_rate']
        
        # Resample if needed
        if orig_sr != self.sr:
            # Resample needs float32
            audio_array = audio_array.astype(np.float32)
            y = librosa.resample(audio_array, orig_sr=orig_sr, target_sr=self.sr)
        else:
            y = audio_array.astype(np.float32)
            
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

def get_hf_dataloaders(hf_id, batch_size=32):
    # Hugging Face datasets usually have 'train', 'validation', 'test' splits or just 'train'.
    # AbstractTTS/IEMOCAP structure: checking...
    # If standard splits exist:
    try:
        train_ds = IEMOCAPHFDataset(hf_id, split="train")
        # Try to load validation/test if exists, else split train
        try:
            val_ds = IEMOCAPHFDataset(hf_id, split="validation")
        except:
             # If no validation split, manually split train? 
             # For simplicity now, let's just assume valid split or use train for both if simple test.
             print("No validation split found. Using subset of train or returning None.")
             val_ds = None
             
        test_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds else None
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        
        return train_loader, test_loader
    except Exception as e:
        print(f"Dataset load error: {e}")
        return None, None
