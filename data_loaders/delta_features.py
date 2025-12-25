"""
Delta Feature Extraction for Speech Emotion Recognition

Instead of replicating single-channel spectrograms to 3 channels,
we compute meaningful delta features:
    - Channel 1: Original spectrogram
    - Channel 2: Delta (1st derivative) - captures temporal dynamics
    - Channel 3: Delta-Delta (2nd derivative) - captures acceleration

This is crucial for emotion recognition as prosody (pitch, energy dynamics)
is a key indicator of emotional state.
"""

import numpy as np
import cv2
import librosa


def compute_delta_features(spectrogram, order=2):
    """
    Compute delta and delta-delta features from a spectrogram.
    
    Args:
        spectrogram: 2D numpy array [freq_bins, time_frames]
        order: Maximum order of derivatives (1 for delta only, 2 for delta-delta)
        
    Returns:
        stacked: 3D numpy array [3, freq_bins, time_frames]
                 - [0]: Original spectrogram
                 - [1]: Delta (1st derivative)
                 - [2]: Delta-delta (2nd derivative)
    """
    # Ensure spectrogram is 2D
    if spectrogram.ndim != 2:
        raise ValueError(f"Expected 2D spectrogram, got {spectrogram.ndim}D")
    
    # Original
    original = spectrogram
    
    # Delta: first order derivative using librosa
    # width=9 is standard for speech features (like MFCCs)
    delta = librosa.feature.delta(spectrogram, order=1, width=9)
    
    if order >= 2:
        # Delta-Delta: second order derivative
        delta_delta = librosa.feature.delta(spectrogram, order=2, width=9)
    else:
        delta_delta = np.zeros_like(spectrogram)
    
    # Stack: [3, freq_bins, time_frames]
    stacked = np.stack([original, delta, delta_delta], axis=0)
    
    return stacked


def resize_normalize_with_delta(spectrogram, target_size, imagenet_norm=True):
    """
    Compute delta features, resize, and normalize for pretrained vision models.
    
    Args:
        spectrogram: 2D numpy array [freq_bins, time_frames]
        target_size: (height, width) tuple
        imagenet_norm: If True, normalize with ImageNet mean/std
        
    Returns:
        Tensor-ready numpy array [3, H, W]
    """
    # Compute delta features: [3, freq_bins, time_frames]
    delta_features = compute_delta_features(spectrogram, order=2)
    
    # Normalize each channel to [0, 1] independently
    normalized = np.zeros_like(delta_features, dtype=np.float32)
    for i in range(3):
        channel = delta_features[i]
        c_min, c_max = channel.min(), channel.max()
        if c_max - c_min > 1e-8:
            normalized[i] = (channel - c_min) / (c_max - c_min)
        else:
            normalized[i] = np.zeros_like(channel)
    
    # Resize each channel
    resized = np.zeros((3, target_size[0], target_size[1]), dtype=np.float32)
    for i in range(3):
        resized[i] = cv2.resize(normalized[i], (target_size[1], target_size[0]))
    
    # ImageNet normalization
    if imagenet_norm:
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        resized = (resized - mean) / std
    
    return resized


def extract_cqt_with_delta(y, sr, target_size):
    """
    Extract CQT spectrogram with delta features.
    
    Args:
        y: Audio signal
        sr: Sample rate
        target_size: (H, W) tuple
        
    Returns:
        Tensor-ready numpy array [3, H, W]
    """
    # Compute CQT
    cqt = librosa.cqt(y, sr=sr)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    
    # Compute delta features and normalize
    return resize_normalize_with_delta(cqt_db, target_size)


def extract_mel_with_delta(y, sr, n_fft, hop_length, target_size):
    """
    Extract Mel spectrogram with delta features.
    
    Args:
        y: Audio signal
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length
        target_size: (H, W) tuple
        
    Returns:
        Tensor-ready numpy array [3, H, W]
    """
    # Compute Mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Compute delta features and normalize
    return resize_normalize_with_delta(mel_db, target_size)


# ==================== Test ====================

if __name__ == "__main__":
    print("Testing Delta Feature Extraction...")
    
    # Create dummy spectrogram
    spec = np.random.randn(128, 200).astype(np.float32)
    
    # Test delta computation
    delta_features = compute_delta_features(spec)
    print(f"Input shape: {spec.shape}")
    print(f"Delta features shape: {delta_features.shape}")
    
    # Test full pipeline
    target_size = (244, 244)
    result = resize_normalize_with_delta(spec, target_size)
    print(f"Final output shape: {result.shape}")
    print(f"Channel 0 (original) range: [{result[0].min():.3f}, {result[0].max():.3f}]")
    print(f"Channel 1 (delta) range: [{result[1].min():.3f}, {result[1].max():.3f}]")
    print(f"Channel 2 (delta-delta) range: [{result[2].min():.3f}, {result[2].max():.3f}]")
    
    # Test with real audio
    print("\nTesting with synthetic audio...")
    duration = 3.0
    sr = 44100
    t = np.linspace(0, duration, int(sr * duration))
    y = np.sin(2 * np.pi * 440 * t) * np.exp(-t)  # Decaying sine wave
    
    cqt_result = extract_cqt_with_delta(y, sr, target_size)
    mel_result = extract_mel_with_delta(y, sr, n_fft=4096, hop_length=256, target_size=target_size)
    
    print(f"CQT with delta shape: {cqt_result.shape}")
    print(f"Mel with delta shape: {mel_result.shape}")
    
    print("\n✅ All tests passed!")
