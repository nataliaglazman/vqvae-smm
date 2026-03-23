# Hierarchical VQ-VAE-2 for 3D Brain MRI

## 1. Model Architecture

### Overview

The model is a **3-level hierarchical VQ-VAE-2** adapted for 3D volumetric brain MRI. Each level captures features at a different spatial resolution: Level 0 (finest) encodes local structural detail, while Level 2 (coarsest) captures global, low-frequency anatomy. The architecture is implemented in `vqvae2.py`.

### Encoder

Each level has a dedicated 3D encoder composed of:

- **Strided convolution blocks** (`Conv3d` with kernel 4, stride 2, padding 1) for spatial downsampling, followed by `BatchNorm3d` and `ReLU`
- A **residual stack** of ReZero blocks (see below) for feature refinement
- Channel progression: `in_channels` &rarr; `hidden_channels // 2` &rarr; `hidden_channels`

Level 0 takes the raw input (1 channel); subsequent levels take the preceding encoder's output (64 channels). With the default `scaling_rates = [2, 2, 2]`, total spatial downsampling is 8&times; per axis.

### Vector Quantisation (CodeLayer)

Each encoder's output is quantised by an EMA-updated codebook:

| Parameter | Value |
|-----------|-------|
| Embedding dimension | 32 |
| Codebook entries | 384 per level |
| EMA decay | 0.99 |
| Dead-code threshold | 1.0 |
| Commitment weight | 0.25 |

**Quantisation procedure.** A 1&times;1&times;1 convolution projects the encoder output (optionally concatenated with the coarser level's decoder output) to `embed_dim = 32`. L2 distances to all codebook vectors are computed and the nearest entry is selected. A straight-through estimator copies gradients from the decoder input to the encoder output.

**EMA codebook updates.** During training, codebook vectors are updated via exponential moving averages of the encoder outputs assigned to each entry. The implementation uses `bincount` + `scatter_add_` rather than one-hot matrices, reducing memory overhead from ~11 GB to near zero at batch size 8 and 1 mm spacing.

**Dead-code reset.** Codes with `cluster_size < 1.0` are detected each forward pass and re-initialised to randomly sampled encoder outputs, ensuring the full codebook remains active.

**Entropy regularisation (optional).** An entropy penalty on the soft assignment distribution encourages uniform codebook usage (`entropy_weight = 0.1`).

### Decoder

Each level has a 3D decoder composed of:

- An initial `Conv3d` (kernel 3) followed by a residual stack
- **Transposed convolution blocks** (`ConvTranspose3d`, kernel 4, stride 2) for upsampling, each followed by `BatchNorm3d` and `ReLU`
- A final `Conv3d` (kernel 3) projection to the output channel dimension

The Level 0 decoder receives the concatenation of all levels' quantised codes (upscaled to match its spatial resolution) and outputs the single-channel reconstruction. Intermediate-level decoders output `embed_dim`-dimensional features that condition finer-level codebook inputs.

### Upscaler

Learned `ConvTranspose3d` modules upscale coarser-level code embeddings to match finer-level spatial dimensions, enabling hierarchical conditioning across the decoder.

### ReZero Residual Block

Each residual block uses the ReZero initialisation scheme: the residual branch output is multiplied by a learnable scalar &alpha; (initialised to 0) before being added to the skip connection. The branch itself consists of two `Conv3d`&ndash;`BatchNorm3d` layers with `ReLU` activation between them (no final ReLU, allowing the residual to both add and subtract values). Optional gradient checkpointing (`torch.utils.checkpoint`) reduces peak memory at the cost of recomputation.

### Default Hyperparameters

| Parameter | Default |
|-----------|---------|
| Input channels | 1 |
| Hidden channels | 64 |
| Residual channels | 32 |
| Residual blocks per level | 2 |
| Number of levels | 3 |
| Codebook entries per level | 384 |
| Embedding dimension | 32 |
| Scaling rates | [2, 2, 2] |

---

## 2. Loss Function

Training uses a composite loss (`BaselineLoss` in `loss.py`) with four reconstruction terms plus a VQ commitment term.

### 2.1 Pixel Loss (L1 / MAE)

$$\mathcal{L}_\text{pixel} = \| \mathbf{x} - \hat{\mathbf{x}} \|_1$$

Weight: `pixel_factor = 1.0`.

### 2.2 Spectral / FFT Loss

The absolute magnitudes of the 3D real FFT (with orthonormal normalisation) of the ground truth and reconstruction are compared via MSE:

$$\mathcal{L}_\text{FFT} = \| \, |\mathcal{F}(\mathbf{x})| - |\mathcal{F}(\hat{\mathbf{x}})| \, \|_2^2$$

Inputs are normalised to [0, 1] using per-sample min-max statistics (derived from the ground truth) before the FFT. For volumes exceeding 128&sup3; voxels, both tensors are downsampled with trilinear interpolation to limit memory consumption.

Weight: `fft_factor = 10.0`.

### 2.3 Perceptual Loss (LPIPS)

A frozen SqueezeNet-based LPIPS network measures perceptual similarity. Because LPIPS operates on 2D images, slices are extracted from three orthogonal planes:

- 32 slices from each of the sagittal, coronal, and axial orientations (96 slices total)
- Each slice is resized to 96 &times; 96 and replicated to 3 channels
- A single batched LPIPS forward pass computes per-slice distances
- The final perceptual loss is the mean across all orientations

Inputs are clamped to [-1, 1] to match the pretrained backbone's expected range.

Weight: `perceptual_factor = 1.0`.

### 2.4 Gradient Domain Loss (GDL)

L1 differences of first-order spatial gradients (finite differences) along all three axes:

$$\mathcal{L}_\text{GDL} = \sum_{d \in \{x, y, z\}} \| \nabla_d \mathbf{x} - \nabla_d \hat{\mathbf{x}} \|_1$$

This penalises blurred or shifted edges in the reconstruction.

Weight: `gdl_factor = 1.0`.

### 2.5 VQ Commitment Loss

Per-level commitment losses from the codebook (MSE between encoder output and the selected codebook vector) are weighted and summed:

$$\mathcal{L}_\text{VQ} = \beta \sum_{\ell} \| \text{sg}[\mathbf{e}_\ell] - \mathbf{z}_\ell \|_2^2$$

Weight: `commitment_weight (beta) = 0.25`.

### Total Loss

$$\mathcal{L} = \underbrace{(\mathcal{L}_\text{pixel} + \mathcal{L}_\text{FFT} + \mathcal{L}_\text{perc} + \mathcal{L}_\text{GDL})}_{\text{scaled by } \texttt{scale\_recon\_loss}} + \; \mathcal{L}_\text{VQ}$$

The reconstruction component can be independently scaled (default 1.0) without affecting the VQ commitment cost.

---

## 3. Data Pipeline

### Dataset

Training data consists of T1-weighted brain MRI scans (NIfTI format) registered to a common template. A CSV file maps each subject to a diagnostic group (AD, CN, or MCI). Data loading and subject selection are handled in `utils.py`.

### Preprocessing (Deterministic, Cached)

All deterministic transforms are applied once and cached in RAM (default 25% of training data):

1. **Load** NIfTI files
2. **Ensure 3D** (squeeze singleton dimensions or extract first volume from 4D)
3. **Channel-first** layout: (C, D, H, W)
4. **Resample** to isotropic voxel spacing (default 2.0 mm for fast experiments, 1.0 mm for full resolution) using MONAI's `Spacingd`
5. **Orient** to RAS (Right-Anterior-Superior)
6. **Pad or crop** to a fixed spatial size derived from the spacing (e.g., 91 &times; 109 &times; 91 at 2 mm)
7. **Intensity normalisation**: zero-mean, unit-standard-deviation (computed over nonzero voxels, per channel)

### Augmentation (Random, Per Epoch)

Applied on-the-fly during training:

| Transform | Probability | Parameters |
|-----------|-------------|------------|
| `RandAffined` | 0.5 | Rotation &plusmn;0.05 rad (3 axes), shear &plusmn;0.05 (6 axis-pairs), scale &plusmn;5%, bilinear interpolation |
| `RandShiftIntensityd` | 0.2 | Uniform intensity offset in [-0.1, 0.1] |

### Train / Validation Split

Stratified by diagnostic group with a default 85/15 split (`val_size = 0.15`).

---

## 4. Training

### Optimiser

**AdamW** with a default learning rate of 1 &times; 10&supmin;&sup5;.

### Learning Rate Schedule

A two-phase schedule:

1. **Linear warmup** (500 steps): LR ramps from 1% to 100% of the target
2. **Cosine annealing**: LR decays to zero over the remaining steps

### Mixed Precision

Optional automatic mixed precision (AMP) via `torch.amp.autocast` with a `GradScaler`. The codebook quantisation layer is forced to `float32` for numerical stability regardless of AMP state.

### Gradient Handling

- **Gradient clipping**: max norm = 1.0
- **Gradient accumulation**: configurable (default 1 step; effective batch size = batch_size &times; accumulation steps)
- **Gradient checkpointing**: optional in residual blocks to reduce peak memory

### Memory Optimisations

- Channels-last-3D memory format (`torch.channels_last_3d`)
- TF32 enabled on Ampere+ GPUs
- cuDNN benchmarking enabled
- Optional `torch.compile` with the Inductor backend (PyTorch 2.0+)

### Reconstruction Skip

A configurable fraction of training steps (`skip_recon_ratio`) can skip the decoder entirely, computing only the VQ commitment loss. This is useful for warming up the codebook before engaging the full loss.

### Checkpointing & Logging

| Output | Frequency |
|--------|-----------|
| TensorBoard scalars (loss, LR, per-level VQ loss, loss components) | Every 100 steps |
| CSV logs (`train_losses.csv`, `val_losses.csv`) | Every 100 steps / every checkpoint |
| Latest checkpoint (`checkpoint_latest.pt`) | Every 1,000 steps |
| Best checkpoint (`checkpoint_best.pt`) | On validation improvement |
| Final checkpoint (`checkpoint_final.pt`) | End of training |

Default training duration: **300,001 steps**.

---

## 5. Evaluation

### Validation Loss

At each checkpoint step, the full validation set is processed through the composite loss function. The best checkpoint is selected by lowest validation loss.

### Reconstruction Quality

Example original and decoded volumes are saved as NIfTI files at each checkpoint for visual inspection.

### Codebook Utilisation

Two complementary metrics are reported:

1. **EMA-based (training statistics):** Counts codes whose EMA cluster size exceeds the dead-code threshold. Perplexity (exp of usage entropy) indicates how uniformly the codebook is used. Note: dead-code resets can inflate these statistics, potentially reporting 100% utilisation even when many codes are rarely selected during inference.

2. **Inference-based (actual usage):** After validation, all codebook indices from the validation set are collected and actual per-code usage is counted. This gives the ground-truth active code count and perplexity, unaffected by EMA smoothing or dead-code resets.

Both are reported per level. Typical expected behaviour is near-full utilisation at the finest level (Level 0) with somewhat lower utilisation at coarser levels that have smaller spatial maps.

### Additional Analysis Modes (eval.py)

| Mode | Description |
|------|-------------|
| `visualize` | Encoder feature grids, codebook heatmaps, reconstruction comparison |
| `features` | Export per-level feature maps as PNG and NIfTI |
| `rf-analytical` | Theoretical receptive field from kernel sizes and strides |
| `rf-empirical` | Empirical receptive field via gradient back-propagation |

---

## 6. Summary of Architecture

```
Input (1, D, H, W)
  |
  +-- Encoder 0 (1 -> 64 ch, 2x downsample)
  |     |
  |     +-- Encoder 1 (64 -> 64 ch, 2x downsample)
  |     |     |
  |     |     +-- Encoder 2 (64 -> 64 ch, 2x downsample)
  |     |           |
  |     |           +-- Codebook 2 (64 -> 32-dim, 384 entries)   [coarsest]
  |     |           |     |
  |     |           |     +-- Decoder 2 -> conditioning
  |     |           |
  |     |     +-- Codebook 1 (64+32 -> 32-dim, 384 entries)
  |     |           |
  |     |           +-- Decoder 1 -> conditioning
  |     |
  |     +-- Codebook 0 (64+32 -> 32-dim, 384 entries)            [finest]
  |           |
  |           +-- Decoder 0 (32*3=96 ch input -> 1 ch output)
  |
Output (1, D, H, W)   [interpolated to input size if needed]
```

Each decoder receives the concatenation of all levels' quantised code embeddings (upscaled to its spatial resolution). The coarsest level's decoder output conditions the next-finer level's codebook input, creating a top-down information flow that allows finer levels to encode residual detail.
