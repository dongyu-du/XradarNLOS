# X-band Radar Non-Line-of-Sight Imaging

Code for the CVPR paper **"X-band Radar Non-Line-of-Sight Imaging"**.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Stage 1 — Detect LOS & mNLOS points (SwinUNet)

```bash
# Real data
python infer_swin.py --mode real \
    --input_dir  data/real_data/test \
    --output_dir results/real/stage1

# Simulation data
python infer_swin.py --mode sim \
    --input_dir  data/simulated_data/test/RA \
    --gt_dir     data/simulated_data/test/points \
    --file_list  data/data_splits/test_files.txt \
    --output_dir results/sim/stage1
```

### Stage 2 — NLOS reconstruction (ResidualReflectNet)

```bash
# Real data (with GT evaluation and figures)
python infer_nlos.py \
    --input_dir  results/real/stage1 \
    --label_dir  data/real_data/test \
    --output_dir results/real/stage2 \
    --save_figs

# Simulation data
python infer_nlos.py \
    --input_dir   results/sim/stage1 \
    --output_dir  results/sim/stage2 \
    --checkpoint  checkpoints/reflectnet_stage2_simdata.pth \
    --f1_threshold 1.0
```

## Training

```bash
# Stage 1 — fine-tune SwinUNet on real data
python train_swin.py --mode real \
    --label_dir   data/real_data/train \
    --train_list  data/data_splits/exp_train_files.txt \
    --val_list    data/data_splits/exp_test_files.txt \
    --output      checkpoints/swinunet_stage1_realdata.pth \
    --start_ckpt  checkpoints/swinunet_stage1_realdata.pth

# Stage 2 — train ResidualReflectNet on simulation data
python train_nlos.py --mode sim \
    --data_pattern data/simulated_data/train/points \
    --train_list   data/data_splits/train_files.txt \
    --test_list    data/data_splits/test_files.txt \
    --output       checkpoints/reflectnet_stage2_simdata.pth
```

## Citation

```bibtex
@inproceedings{xbandradar_cvpr,
  title     = {X-band Radar Non-Line-of-Sight Imaging},
  booktitle = {CVPR 2026},
}
```
