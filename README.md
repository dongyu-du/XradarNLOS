# X-band NLOS Radar Reconstruction

A two-stage pipeline for **Non-Line-of-Sight (NLOS) reconstruction** using ground-level X-band radar.

Given a Range-Azimuth (RA) spectrogram from a single stationary radar, the system localises objects hidden behind walls.

---

## Pipeline Overview

```
RA Spectrogram (512×256)
        │
        ▼
┌─────────────────────────────┐
│  Stage 1 · SwinUNet         │  infer_swin.py
│  Detects two point types:   │
│    L  – LOS reflections     │
│    G  – mNLOS reflections   │
└──────────────┬──────────────┘
               │  L, G  (x, y) point clouds
               ▼
┌──────────────────────────────────────────┐
│  Stage 2 · Geometry + ResidualReflectNet │  infer_nlos.py
│  1. DBSCAN clusters L → wall segments   │
│  2. RANSAC fits a line to each cluster  │
│  3. Reflect G off wall → T_geom         │
│  4. NN predicts correction Δ            │
│     T_pred = T_geom + Δ                 │
└──────────────────────────────────────────┘
               │
               ▼
     T_pred  – estimated NLOS positions (x, y)
```

---

## Quick Reproduction

To reproduce the full test-set results with a single command:

```bash
cd public/
bash reproduce.sh
```

This runs Stage 1 (SwinUNet) then Stage 2 (NLOS reconstruction) on all 32 real-data test captures and saves results + SVG visualisations to `results/`.

---

## Installation

```bash
pip install -r requirements.txt
```

Tested with Python 3.8+ and PyTorch ≥ 1.9.

---

## Data

```
data/
├── real_data/
│   ├── train/    97 captures (scenes 3–14)  ← exp_train_files.txt
│   └── test/     32 captures (scenes 3–13)  ← exp_test_files.txt
│
├── simulated_data/
│   ├── train/
│   │   ├── points/   1 512 .npz  L/G/T point clouds (MLNs_gt)        ← Stage 2 training
│   │   ├── RA/       1 512 .npy  raw complex RA (512×256)             ← Stage 1 training input
│   │   └── RA_gt/    1 512 .npy  per-pixel heatmap labels (512×256×5) ← Stage 1 training labels
│   ├── val/
│   │   ├── points/     324 .npz  L/G/T point clouds                   ← Stage 2 validation
│   │   ├── RA/         324 .npy  raw complex RA                       ← Stage 1 validation input
│   │   └── RA_gt/      324 .npy  per-pixel heatmap labels             ← Stage 1 validation labels
│   └── test/
│       ├── points/     324 .npz  L/G/T point clouds (MLNs_gt)         ← Stage 2 test
│       ├── RA/         324 .npy  raw complex RA                       ← Stage 1 test input
│       └── RA_gt/      324 .npy  per-pixel heatmap labels             ← Stage 1 test labels
│
└── data_splits/  all .txt split files for reference
```

Each real-data capture contains:
```
scene{N}/capture{M}/
  welch_interpolated_background_subtracted.npz   ← Stage 1 input (key: Pxx, shape 512×256)
  labeled_points/
    los.npy            LOS ground-truth points   (N, 2)
    mirrored_nlos.npy  mNLOS ground-truth points (M, 2)
    nlos.npy           NLOS ground-truth points  (K, 2)
    wall.npy           wall sample points        (W, 2)
  camera.jpg           scene photo
```

> **Note on hardware_config raw RA simulation files:**
> The raw simulation RA arrays (`hardware_config/RAs/`, ~2.1 MB each, ~4.5 GB total) are not included in
> this package due to size. They are only required to run Stage 1 inference on simulated data.
> Contact the authors for access if needed.

---

## Checkpoints

| File | Description |
|------|-------------|
| `swinunet_stage1_realdata_v2.pth` | SwinUNet v2 fine-tuned on real data — **recommended for Stage 1** |
| `swinunet_exp_finetuned_bestF1.pth` | SwinUNet v1 (fallback) |
| `swin_unet_finetuned_real_data_new_split.pth` | SwinUNet trained on a different split |
| `reflectnet_stage2_realdata.pth` | ResidualReflectNet fine-tuned on real data — **recommended for Stage 2** |
| `residual_reflectnet_best_sim.pth` | ResidualReflectNet trained on simulation only |

---

## Inference

All scripts must be run from the `public/` directory so that local package imports (`utils/`, `networks/`) resolve correctly.

### Stage 1 — Detect LOS & mNLOS points

```bash
python infer_swin.py \
    --input_dir   data/real_data/test \
    --output_dir  results/stage1_swinunet \
    --checkpoint  checkpoints/swinunet_stage1_realdata_v2.pth
```

**Input tree:**
```
input_dir/
  scene{N}/
    capture{M}/
      welch_interpolated_background_subtracted.npz   ← key: Pxx  shape (512, 256)
```

**Output tree:**
```
output_dir/
  scene{N}/
    capture{M}.npz    ← keys: L (3, N_los), G (3, M_mnlos)
```

Key arguments:
- `--conf_thres` – peak detection threshold (default `0.15`)
- `--min_dis` – minimum peak separation in pixels (default `3`)

---

### Stage 2 — NLOS reconstruction

```bash
python infer_nlos.py \
    --input_dir   results/stage1_swinunet \
    --label_dir   data/real_data/test \
    --output_dir  results/stage2_nlosrecon \
    --checkpoint  checkpoints/reflectnet_stage2_realdata.pth \
    --save_figs
```

`--label_dir` is only needed for ground-truth evaluation metrics. If omitted, the script still runs and saves predictions.

**Output tree:**
```
output_dir/
  scene{N}/
    capture{M}.npz    ← keys: L, G, T_pred, T_geom  (and T if GT available)
  figs/
    scene{N}/
      capture{M}.svg  ← visualisation (with --save_figs)
```

Key arguments:
- `--use_gt_wall` – replace DBSCAN+RANSAC with labeled wall points (upper-bound evaluation only)
- `--f1_threshold` – matching radius in metres for F1 score (default `2.0`)
- `--save_figs` – save per-capture SVG visualisations

---

## Data Format

### RA spectrogram input
- File: `welch_interpolated_background_subtracted.npz`
- Key: `Pxx`  shape `(512, 256)` — background-subtracted Welch PSD
- Range axis: 512 bins spanning **4.521 – 43.994 m**
- Azimuth axis: 256 bins spanning **−60° to +60°**

### Labeled point files (for evaluation)
```
labeled_points/
  los.npy            shape (N, 2)  – LOS xy coordinates in metres
  mirrored_nlos.npy  shape (M, 2)  – mNLOS xy coordinates in metres
  nlos.npy           shape (K, 2)  – NLOS ground truth xy coordinates
  wall.npy           shape (W, 2)  – wall sample points
```

### Point cloud arrays
- Intermediate and output `.npz` files store point clouds with keys:
  - `L` shape `(3, N)` or `(N, 2)` — LOS points `(x, y, [0])`
  - `G` shape `(3, M)` or `(M, 2)` — mNLOS points
  - `T` shape `(K, 2)` — NLOS ground truth
  - `T_pred` shape `(M, 2)` — NN-corrected NLOS prediction
  - `T_geom` shape `(M, 2)` — geometry-only baseline prediction

---

## Training

### Stage 1 — Fine-tune SwinUNet on real data

```bash
python train_swin.py \
    --label_dir   data/real_data/train \
    --train_list  data/data_splits/exp_train_files.txt \
    --val_list    data/data_splits/exp_test_files.txt  \
    --output      checkpoints/swinunet_finetuned.pth   \
    --start_ckpt  checkpoints/swinunet_stage1_realdata_v2.pth
```

> The `--train_list` file contains paths like `scene8/capture7` relative to `--label_dir`.

### Stage 2 — Train ResidualReflectNet on simulation data

First, run SwinUNet on the simulation training set to produce L/G/T `.npz` files, then:

```bash
python train_nlos.py \
    --data_pattern data/simulated_data/train/points \
    --train_list   data/data_splits/train_files.txt \
    --test_list    data/data_splits/test_files.txt  \
    --output       checkpoints/residual_reflectnet.pth
```

> `data/simulated_data/train/points/` contains pre-labeled `L`, `G`, `T` point clouds (`.npz`) ready for
> training. The raw RA simulation inputs (`data/simulated_data/train/RA/`) are not required for this stage.

---

## Model Architectures

### SwinUNet (Stage 1)
- Backbone: `SwinTransformerSys` — Swin Transformer encoder with skip-connection UNet decoder
- Config: `img_size=512`, `patch_size=4`, `embed_dim=96`, `depths=[2,2,6,2]`, `window_size=8`
- Output: 2-channel heatmap `(B, 2, 512, 256)` — channel 0 = LOS, channel 1 = mNLOS
- Peak detection with sigmoid threshold and `skimage.feature.peak_local_max`

### ResidualReflectNet (Stage 2)
- Inputs per mNLOS point:
  - `G_i` — 2D position of the mNLOS point
  - `L_patch` — k=16 nearest LOS neighbours (attention-pooled context)
  - `S_i` — wall line parameters `(a, b, c)`
  - `is_noise` — flag indicating whether the buddy LOS point is a DBSCAN outlier
- Output: residual correction `Δ ∈ ℝ²` added to the geometric reflection

---

## Citation

If you use this code, please cite our paper (citation info to be added upon publication).
