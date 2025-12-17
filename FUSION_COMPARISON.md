# MaxMViT-MLP với GMU Fusion (GloMER-inspired)

## 📊 So sánh Fusion Strategies

### Original MaxMViT-MLP (Concatenation)
```
CQT → MaxViT → [feat_cqt]
                           → concat → [feat_cqt; feat_mel] → MLP → output
Mel-STFT → MViTv2 → [feat_mel]
```

**Vấn đề:**
- Static fusion: không adapt theo input
- Không kiểm soát modality dominance
- Feature dimension lớn (dim_cqt + dim_mel)

### Improved (GMU Fusion)
```
CQT → MaxViT → [feat_cqt] → tanh projection → z̃_cqt
                                                        → GMU → z_fused → MLP
Mel-STFT → MViTv2 → [feat_mel] → tanh projection → z̃_mel

GMU: g = σ(W[z̃_cqt; z̃_mel])
     z_fused = g ⊙ z̃_cqt + (1-g) ⊙ z̃_mel
```

**Ưu điểm:**
- Dynamic fusion: tự điều chỉnh weight theo input
- Balanced modality contribution
- Smaller feature dimension
- Interpretable (xem gate values)

---

## 🔑 GMU - Cách hoạt động

```python
# Gate vector học cách cân bằng 2 modalities
g = sigmoid(W_gate @ concat(z_cqt, z_mel))  # g ∈ [0, 1]

# Adaptive fusion
z_fused = g * z_cqt + (1 - g) * z_mel

# Interpretation:
# g → 1.0: Model tin CQT/MaxViT path nhiều hơn
# g → 0.0: Model tin Mel-STFT/MViTv2 path nhiều hơn
# g ≈ 0.5: Balanced fusion
```

---

## 📈 Expected Results

| Model | IEMOCAP (paper) | Fusion Type |
|-------|-----------------|-------------|
| MaxMViT-MLP (original) | 68.39% | Concatenation |
| MaxMViT-MLP + GMU | ~70-72% (expected) | Gated Fusion |
| MaxMViT-MLP + GMU + Contrastive | ~72-75% (expected) | Gated + Contrastive |

**Note:** GloMER đạt 82.79% trên IEMOCAP nhưng họ dùng **Text + Audio** (BERT + Wav2Vec).
MaxMViT-MLP dùng **CQT + Mel-STFT** (cùng source audio), nên improvement sẽ modest hơn.

---

## 🛠️ Cách sử dụng

### 1. GMU Fusion (recommended để bắt đầu)
```bash
python train_gmu.py --config configs/iemocap_gmu.yaml
```

### 2. GMU + Contrastive Learning
```bash
python train_gmu.py --config configs/iemocap_gmu_contrastive.yaml
```

### 3. So sánh với Original
```bash
# Original (concat fusion)
python train.py --config configs/iemocap.yaml

# New (GMU fusion)
python train_gmu.py --config configs/iemocap_gmu.yaml
```

---

## 📁 File Structure

```
maxmvit_gmu/
├── model_gmu.py                 # GMU model implementation
├── train_gmu.py                 # Training script with GMU support
├── configs/
│   ├── iemocap_gmu.yaml         # GMU only
│   └── iemocap_gmu_contrastive.yaml  # GMU + Contrastive
└── FUSION_COMPARISON.md         # This file
```

---

## 🔬 Ablation Study Guide

Để so sánh đầy đủ, chạy 3 experiments:

| Experiment | Config | Fusion |
|------------|--------|--------|
| Baseline | `iemocap.yaml` (fix lr=0.02) | concat |
| GMU only | `iemocap_gmu.yaml` | gmu |
| GMU + CL | `iemocap_gmu_contrastive.yaml` | gmu_contrastive |

---

## 🎛️ Hyperparameter Tuning

### GMU
- `fusion_hidden_dim`: None (auto) hoặc 512, 768, 1024

### Contrastive Learning (từ GloMER paper)
- `alpha`: 0.3 (IEMOCAP), 0.5 (ESD), tune trong [0, 1.5]
- `temperature`: 0.07 (standard NT-Xent)

### Learning Rate
- Giữ 0.02 như MaxMViT-MLP paper
- GloMER dùng 1e-4 nhưng họ dùng pretrained BERT/Wav2Vec

---

## 📝 Key Differences: GloMER vs This Implementation

| Aspect | GloMER | This (MaxMViT-MLP + GMU) |
|--------|--------|--------------------------|
| Modalities | Text (BERT) + Audio (Wav2Vec) | CQT (MaxViT) + Mel-STFT (MViTv2) |
| Information source | 2 different sources | Same audio, 2 representations |
| Cross-modal attention | Yes | No (could add) |
| GMU | Yes ✓ | Yes ✓ |
| Contrastive Learning | Yes | Optional ✓ |
| Expected benefit | High (complementary info) | Moderate (redundant info) |

---

## 🚀 Next Steps

1. **Run baseline** với lr=0.02 fix
2. **Run GMU** và compare
3. **Analyze gate values** để hiểu model behavior
4. **Try contrastive** nếu GMU improves
5. **Optional**: Add cross-modal attention (như GloMER)
