# Codebase index — MaxMViT-MLP Speech Emotion Recognition

This repo implements a dual-path SER model that fuses two image backbones (MaxViT + MViTv2) on top of two spectrogram views (CQT + Mel-STFT).

## Top-level layout

- `train.py`: training entrypoint (reads YAML config, builds dataloaders + model, trains, checkpoints).
- `train_ablation.py`: runs multiple Fusion+Head variants for ablation and writes comparison tables (CSV/JSON/MD).
- `model.py`: `MaxMViT_MLP` model definition + `get_optimizer()` (paper-style optimizer split).
- `configs/`: experiment configs (`iemocap.yaml`, `ravdess.yaml`).
- `data_loaders/`: dataset loaders (HuggingFace and local), selected via a factory.
- `utils.py`: config loading, logging setup, seeding.
- `dataset.py`: a standalone dataset for “list of audio paths + labels” (not used by `train.py`).

Artifacts (created at runtime):

- `.cache_spectrograms/`: on-disk cache of precomputed spectrogram tensors (`*.pkl`) for HF datasets.
- `logs/<experiment_name>/`: training logs.
- `checkpoints/<experiment_name>/`: top-k checkpoints saved during training.

## Main call graph

1. `train.py:train(config_path)`
2. `utils.py:load_config()` → reads YAML in `configs/`
3. `data_loaders.get_dataloaders(config)` → routes by `dataset.name`
   - `data_loaders/ravdess.py:get_ravdess_dataloaders(...)`
   - `data_loaders/iemocap_hf.py:get_hf_dataloaders(...)`
   - `data_loaders/iemocap_local.py:get_iemocap_dataloaders(...)` (currently not wired in the factory)
4. `model.py:MaxMViT_MLP(num_classes=...)`
5. `model.py:get_optimizer(model, lr=...)` → returns 2 optimizers
6. Train loop:
   - forward: `logits = model(cqt, mel)`
   - loss: `CrossEntropyLoss(logits, label)`
   - save checkpoint each epoch; keep top-3 by val accuracy

## Data flow and tensor shapes (what the model expects)

The training loaders in `data_loaders/` return:

- `cqt`: `float32` tensor shaped `[3, H, W]` (3-channel, ImageNet-normalized)
- `mel`: `float32` tensor shaped `[3, H, W]` (3-channel, ImageNet-normalized)
- `label`: `long` tensor shaped `[]` (scalar)

The `MaxMViT_MLP.forward(cqt, mel)` accepts either 1-channel or 3-channel inputs:

- If channel dimension is 1, it repeats to 3 channels.
- If spatial size is not `224×224`, it resizes to `224×224` with bilinear interpolation.
- It then extracts features with:
  - `timm.create_model('maxvit_base_tf_224', pretrained=True, num_classes=0)` → `feat_maxvit` `[B, 768]`
  - `timm.create_model('mvitv2_base', pretrained=True, num_classes=0)` → `feat_mvitv2` `[B, 768]`
- Fusion: `torch.cat([feat_maxvit, feat_mvitv2], dim=1)` → `[B, 1536]`
- MLP head: `Linear(1536→512) → BN → Dropout → ReLU → Linear(512→num_classes)`

## Key “knobs” to modify

- Change dataset: `configs/*.yaml` (`dataset.name` + `dataset.args`) and `data_loaders/factory.py`.
- Change classes: `configs/*.yaml` → `model.num_classes`.
- Change spectrogram params: `data_loaders/*.py` (`n_fft=4096`, `hop_length=256`, `librosa.cqt(...)`).
- Change fusion/head: `model.py` (`fused = cat(...)`, `self.mlp = nn.Sequential(...)`).
  - Supported `fusion`: `concat`, `gated_scalar`, `gated_channel`, `attn2token`, `hadamard`, `lowrank_bilinear`
  - Supported `head`: `mlp_bn`, `mlp_ln`, `mlp_residual_ln`

## Notes / gotchas

- `configs/*.yaml` set `training.lr: 0.0002` while the README notes the paper value is `0.02`.
- `dataset.py` produces 1-channel spectrogram tensors without ImageNet normalization; it’s fine for quick experiments, but it’s not the same preprocessing as `data_loaders/*_hf.py`.
