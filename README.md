# MaxMViT-MLP: Speech Emotion Recognition

[![Paper](https://img.shields.io/badge/Paper-IEEE%20Access-blue)](https://ieeexplore.ieee.org/document/XXXXXX)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org)

Implementation of **MaxMViT-MLP: Multiaxis and Multiscale Vision Transformers Fusion Network for Speech Emotion Recognition**.

---

## 📐 Model Architecture

The model uses a **dual-path architecture** that processes two types of spectrograms in parallel and fuses them for classification.

```
┌─────────────────────────────────────────────────────────────────┐
│                      INPUT: Raw Audio                           │
└─────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
        ┌───────────────┐               ┌───────────────┐
        │   CQT Transform│               │ Mel-STFT     │
        │   (librosa.cqt)│               │ (n_fft=4096, │
        │                │               │  hop=256)    │
        └───────────────┘               └───────────────┘
                │                               │
                ▼                               ▼
        ┌───────────────┐               ┌───────────────┐
        │ Resize to     │               │ Resize to     │
        │ 244×244 → 224 │               │ 244×244 → 224 │
        │ (3 channels)  │               │ (3 channels)  │
        └───────────────┘               └───────────────┘
                │                               │
                ▼                               ▼
        ┌───────────────┐               ┌───────────────┐
        │   MaxViT      │               │   MViTv2      │
        │   (Base)      │               │   (Base)      │
        │   768 dims    │               │   768 dims    │
        └───────────────┘               └───────────────┘
                │                               │
                └───────────────┬───────────────┘
                                ▼
                        ┌───────────────┐
                        │ Concatenation │
                        │ (1536 dims)   │
                        └───────────────┘
                                │
                                ▼
                        ┌───────────────┐
                        │   MLP Head    │
                        │ (see below)   │
                        └───────────────┘
                                │
                                ▼
                        ┌───────────────┐
                        │   Softmax     │
                        │   (N classes) │
                        └───────────────┘
```

---

## 🔬 Layer-by-Layer Breakdown

### 1. Input Preprocessing (in `data_loaders/*.py`)

| Step | Operation | Parameters | Output Shape |
|------|-----------|------------|--------------|
| 1.1 | Load Audio | `sr=44100` | `(samples,)` |
| 1.2 | CQT Transform | `librosa.cqt(y, sr=44100)` | `(84, T)` |
| 1.3 | Mel-STFT | `n_fft=4096, hop_length=256` | `(128, T)` |
| 1.4 | Resize | `cv2.resize(spec, (244, 244))` | `(244, 244)` |
| 1.5 | 3-Channel Stack | `np.stack([spec]*3)` | `(3, 244, 244)` |
| 1.6 | ImageNet Norm | `mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]` | `(3, 244, 244)` |

> **Note**: Paper specifies 244×244, but the model internally resizes to 224×224 due to MaxViT window size constraints (must be divisible by 7).

---

### 2. Path 1: CQT → MaxViT (in `model.py`)

**Model**: `timm.create_model('maxvit_base_tf_224', pretrained=True, num_classes=0)`

| Block | Layer Type | Details | Output Shape |
|-------|------------|---------|--------------|
| Stem | Conv2d + Norm | `3→96, k=3, s=2` | `(96, 112, 112)` |
| Stage 1 | MBConv × 2 | SE-Attention, Expansion=4 | `(96, 112, 112)` |
| Stage 2 | MBConv × 6 | Downsample + Grid Attention | `(192, 56, 56)` |
| Stage 3 | MBConv × 14 | Block Attention + Grid Attention | `(384, 28, 28)` |
| Stage 4 | MBConv × 2 | Multi-axis Self-Attention | `(768, 14, 14)` |
| Head | Global Avg Pool | `AdaptiveAvgPool2d(1)` | `(768,)` |

**Key Components**:

- **MBConv**: Mobile Inverted Bottleneck with Squeeze-and-Excitation
- **Multi-Axis Attention**: Combines Block (local window) and Grid (dilated) attention
- **Window Size**: 7×7 (reason for 224×224 requirement)

**Total Parameters**: ~119M

---

### 3. Path 2: Mel-STFT → MViTv2 (in `model.py`)

**Model**: `timm.create_model('mvitv2_base', pretrained=True, num_classes=0)`

| Block | Layer Type | Details | Output Shape |
|-------|------------|---------|--------------|
| Patch Embed | Conv2d | `3→96, k=7, s=4` | `(96, 56, 56)` |
| Stage 1 | MViT Block × 2 | Pool Q, Relative Pos Bias | `(96, 56, 56)` |
| Stage 2 | MViT Block × 3 | Downsample, Multi-Head SA | `(192, 28, 28)` |
| Stage 3 | MViT Block × 16 | Pooling Attention | `(384, 14, 14)` |
| Stage 4 | MViT Block × 3 | Final Features | `(768, 7, 7)` |
| Head | Global Avg Pool | `AdaptiveAvgPool2d(1)` | `(768,)` |

**Key Components**:

- **Pooled Q-K-V Attention**: Reduces computation by pooling queries
- **Decomposed Relative Position**: Efficient positional encoding
- **Multiscale Feature Hierarchy**: 4 stages with increasing channel depth

**Total Parameters**: ~52M

---

### 4. Feature Fusion (in `model.py:forward()`)

```python
# Concatenate features from both paths
fused = torch.cat((feat_maxvit, feat_mvitv2), dim=1)  # [B, 1536]
```

| Operation | Input Shape | Output Shape |
|-----------|-------------|--------------|
| MaxViT Output | `(B, 768)` | - |
| MViTv2 Output | `(B, 768)` | - |
| **Concatenation** | - | `(B, 1536)` |

---

### 5. MLP Classification Head (in `model.py`)

```python
self.mlp = nn.Sequential(
    nn.Linear(1536, 512),     # Dense Layer
    nn.BatchNorm1d(512),      # Batch Normalization
    nn.Dropout(0.2),          # Dropout (20%)
    nn.ReLU(),                # Activation
    nn.Linear(512, num_classes)  # Classification Layer
)
```

| Layer | Type | Input → Output | Parameters |
|-------|------|----------------|------------|
| 1 | `nn.Linear` | `1536 → 512` | 786,944 |
| 2 | `nn.BatchNorm1d` | `512 → 512` | 1,024 |
| 3 | `nn.Dropout` | `512 → 512` (p=0.2) | 0 |
| 4 | `nn.ReLU` | `512 → 512` | 0 |
| 5 | `nn.Linear` | `512 → N` | 512×N + N |

> **For IEMOCAP (4 classes)**: Final layer = `512 → 4` = 2,052 params
> **For RAVDESS (8 classes)**: Final layer = `512 → 8` = 4,104 params

---

## ⚙️ Training Configuration

### Optimizers (from Paper Table 2)

| Component | Optimizer | Learning Rate |
|-----------|-----------|---------------|
| MaxViT | Adam | 0.02 |
| MViTv2 | RAdam | 0.02 |
| MLP Head | Adam | 0.02 |

### Learning Rate Scheduler

```yaml
scheduler:
  type: ReduceLROnPlateau
  factor: 0.1
  patience: 2
  min_lr: 1e-6
```

### Training Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Batch Size | 16 | Paper |
| Epochs | 50 | Paper |
| Early Stopping | 5 epochs | Implementation |
| Gradient Clipping | max_norm=1.0 | Implementation |
| Random Seed | 42 | Reproducibility |

---

## 📊 Datasets

### IEMOCAP Configuration

```yaml
dataset:
  name: "iemocap"
  args:
    hf_id: "AbstractTTS/IEMOCAP"
    batch_size: 16
    num_workers: 16
    target_size: [244, 244]
model:
  num_classes: 4  # neu, hap, ang, sad
```

**Emotion Mapping**:

| Raw Label | Mapped Label | Class Index |
|-----------|--------------|-------------|
| neutral | neu | 0 |
| happy | hap | 1 |
| angry | ang | 2 |
| sad | sad | 3 |

> **Note**: `excited` is NOT merged into `happy` in this implementation (user preference).

### RAVDESS Configuration

```yaml
dataset:
  name: "ravdess"
  args:
    hf_id: "TwinkStart/RAVDESS"
    batch_size: 16
    num_workers: 16
    target_size: [244, 244]
model:
  num_classes: 8
```

**Emotion Classes**:
`neutral`, `calm`, `happy`, `sad`, `angry`, `fear`, `disgust`, `surprise`

---

## 🚀 Quick Start

### Training

```bash
# IEMOCAP
python train.py --config configs/iemocap.yaml

# RAVDESS
python train.py --config configs/ravdess.yaml
```

### Project Structure

```
MaxMViT-MLP-SER/
├── configs/
│   ├── iemocap.yaml      # IEMOCAP training config
│   └── ravdess.yaml      # RAVDESS training config
├── data_loaders/
│   ├── __init__.py       # Exports get_dataloaders
│   ├── factory.py        # Dataset factory pattern
│   ├── iemocap_hf.py     # HuggingFace IEMOCAP loader
│   ├── iemocap_local.py  # Local IEMOCAP loader
│   └── ravdess.py        # RAVDESS loader
├── checkpoints/          # Saved models (top-3)
├── logs/                 # Training logs
├── model.py              # MaxMViT-MLP architecture
├── train.py              # Training script
├── utils.py              # Config, logging, seeding
└── requirements.txt
```

---

## 🔧 Key Implementation Details

### 1. Input Size Handling

The paper specifies **244×244**, but:

- MaxViT requires input divisible by **32** and window size **7**
- 244 ÷ 7 = 34.86 (not integer) → **CRASH**
- Solution: Config uses 244×244, model internally resizes to 224×224

```python
# model.py:forward()
if cqt.shape[-1] != 224:
    cqt = F.interpolate(cqt, size=(224, 224), mode='bilinear')
```

### 2. Short Audio Padding

Some audio files are shorter than `n_fft=4096` samples, causing librosa warnings.

```python
# data_loaders/ravdess.py and iemocap_hf.py
if len(y) < self.n_fft:
    padding = self.n_fft - len(y) + 1
    y = np.pad(y, (0, padding), mode='constant')
```

### 3. BatchNorm with Small Batches

Last training batch might have only 1 sample, breaking BatchNorm.

```python
# data_loaders/iemocap_hf.py
train_loader = DataLoader(..., drop_last=True)
```

---

## 📝 Improvement Ideas

Here are potential areas for model improvement (for your paper):

| Area | Current | Possible Improvement |
|------|---------|---------------------|
| **Backbone** | MaxViT-Base + MViTv2-Base | Try Small variants, or newer ViT (DINOv2, EVA) |
| **Fusion** | Simple Concatenation | Attention-based fusion, Bilinear pooling |
| **MLP Head** | 1 hidden layer (512) | 2-3 layers, or Transformer decoder |
| **Spectrogram** | CQT + Mel-STFT | Add MFCC, Log-Mel, or learnable filterbanks |
| **Augmentation** | None | SpecAugment, Mixup, Time masking |
| **Pre-training** | ImageNet | Wav2Vec2.0, HuBERT audio pre-training |
| **Loss** | CrossEntropy | Focal Loss, Label Smoothing |
| **Input Size** | 224×224 (forced) | Use flexible ViT (ViT-FlexPatch) |

---

## 📚 References

- **MaxViT**: [arxiv.org/abs/2204.01697](https://arxiv.org/abs/2204.01697)
- **MViTv2**: [arxiv.org/abs/2112.01526](https://arxiv.org/abs/2112.01526)
- **IEMOCAP**: [sail.usc.edu/iemocap](https://sail.usc.edu/iemocap/)
- **RAVDESS**: [zenodo.org/record/1188976](https://zenodo.org/record/1188976)

---

## 📄 License

MIT License - See LICENSE file for details.
