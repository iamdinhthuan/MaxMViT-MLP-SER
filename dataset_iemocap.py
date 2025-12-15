import os
import torch
import glob
import re
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import cv2

class IEMOCAPDataset(Dataset):
    def __init__(self, root_dir, sessions=None, target_classes=['neu', 'hap', 'ang', 'sad'], sr=44100, target_size=(244, 244)):
        """
        Dataset class for IEMOCAP.
        
        Args:
            root_dir (str): Root directory of IEMOCAP dataset (containing Session1, Session2, etc.)
            sessions (list): List of sessions to include (e.g., ['Session1', 'Session2']). If None, find all.
            target_classes (list): List of emotions to classify. 'exc' is usually merged with 'hap'.
            sr (int): Sampling rate to convert audio to.
            target_size (tuple): Spec image size.
        """
        self.root_dir = root_dir
        self.sr = sr
        self.target_size = target_size
        self.target_classes = target_classes
        self.class_map = {c: i for i, c in enumerate(target_classes)}
        
        # Audio params
        self.n_fft = 4096
        self.hop_length = 256

        if sessions is None:
            self.sessions = [d for d in os.listdir(root_dir) if d.startswith('Session') and os.path.isdir(os.path.join(root_dir, d))]
        else:
            self.sessions = sessions
            
        self.file_paths = [] # List of (wav_path, label_idx)
        
        self._load_data()
        
    def _load_data(self):
        for season in self.sessions:
            dialog_dir = os.path.join(self.root_dir, season, 'dialog')
            emo_dir = os.path.join(dialog_dir, 'EmoEvaluation')
            wav_dir = os.path.join(dialog_dir, 'wav')
            
            # Iterate through EmoEvaluation files
            emo_files = glob.glob(os.path.join(emo_dir, '*.txt'))
            
            for emo_file in emo_files:
                with open(emo_file, 'r') as f:
                    lines = f.readlines()
                    
                for line in lines:
                    # Regex to parse the line: [START - END] TURN_ID EMOTION [V, A, D]
                    # Example: [6.2901 - 8.2357]	Ses01F_impro01_F000	neu	[2.5000, 2.5000, 2.5000]
                    parts = line.strip().split('\t')
                    if len(parts) >= 3 and parts[0].startswith('['):
                        turn_id = parts[1]
                        emotion = parts[2]
                        
                        # Merge 'exc' to 'hap'
                        if emotion == 'exc':
                            emotion = 'hap'
                            
                        if emotion in self.target_classes:
                            wav_path = os.path.join(wav_dir, f"{turn_id}.wav")
                            if os.path.exists(wav_path):
                                self.file_paths.append((wav_path, self.class_map[emotion]))
                                
        print(f"Loaded {len(self.file_paths)} samples from {len(self.sessions)} sessions.")

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        path, label = self.file_paths[idx]
        
        # Load audio
        # Load audio using soundfile
        import soundfile as sf
        y, orig_sr = sf.read(path)
        
        # IEMOCAP is typically 16kHz, but paper used 44.1kHz. So we resample.
        if orig_sr != self.sr:
             y = librosa.resample(y, orig_sr=orig_sr, target_sr=self.sr)
             sr = self.sr
        else:
             sr = orig_sr

        # Ensure mono
        if len(y.shape) > 1:
            y = np.mean(y, axis=1) # Soundfile returns (samples, channels)
        
        # --- Preprocessing same as SERDataset ---
        
        # CQT
        cqt = librosa.cqt(y, sr=sr)
        cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
        
        # Mel-STFT
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Resize & Normalize
        cqt_img = self._resize_normalize(cqt_db)
        mel_img = self._resize_normalize(mel_db)
        
        cqt_tensor = torch.tensor(cqt_img, dtype=torch.float32).unsqueeze(0)
        mel_tensor = torch.tensor(mel_img, dtype=torch.float32).unsqueeze(0)
        
        return cqt_tensor, mel_tensor, torch.tensor(label, dtype=torch.long)

    def _resize_normalize(self, spec):
        spec_min = spec.min()
        spec_max = spec.max()
        spec_norm = (spec - spec_min) / (spec_max - spec_min + 1e-8)
        spec_resized = cv2.resize(spec_norm, (self.target_size[1], self.target_size[0]))
        return spec_resized

def get_iemocap_dataloaders(root_dir, test_session='Session5', batch_size=32, num_workers=4):
    """
    Standard Leave-One-Session-Out split.
    Typically Session 1-4 Train, Session 5 Test.
    """
    all_sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
    if not os.path.exists(os.path.join(root_dir, 'Session5')):
         # Fallback for mock data if Session5 doesn't exist
         all_sessions = [d for d in os.listdir(root_dir) if d.startswith('Session')]
         if not all_sessions:
             raise ValueError("No Session directories found in " + root_dir)
         if len(all_sessions) > 1:
            test_session = all_sessions[-1]
         else:
             test_session = None # Train on all if only 1 session

    train_sessions = [s for s in all_sessions if s != test_session]
    test_sessions = [test_session] if test_session else []
    
    print(f"Train Sessions: {train_sessions}")
    print(f"Test Sessions: {test_sessions}")
    
    train_dataset = IEMOCAPDataset(root_dir, sessions=train_sessions)
    if test_sessions:
        test_dataset = IEMOCAPDataset(root_dir, sessions=test_sessions)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        test_loader = None
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    return train_loader, test_loader
