# X-Band Radar Non-Line-of-Sight Imaging

<div align="center">

### CVPR 2026

**Dongyu Du\*, Mingkun Zhao\*, Yutong Yang, Dominik Scheuble, Xiaolong Huang, Zijian Shao, Mario Bijelic, Kaushik Sengupta, Felix Heide**

*\* Equal contribution*

[![Project Page](https://img.shields.io/badge/Project-Page-blue?style=flat-square)](https://princeton-computational-imaging.github.io/XBandNlos/)
[![Paper](https://img.shields.io/badge/Paper-PDF-red?style=flat-square)](https://princeton-computational-imaging.github.io/XBandNlos/nlos_assets/CVPR2026_NLOS_X_band_radar.pdf)
[![Supplementary](https://img.shields.io/badge/Supplementary-PDF-orange?style=flat-square)](https://princeton-computational-imaging.github.io/XBandNlos/nlos_assets/CVPR2026_NLOS_X_band_radar_Supp.pdf)
[![Video](https://img.shields.io/badge/Video-YouTube-ff0000?style=flat-square&logo=youtube)](https://www.youtube.com/watch?v=eW8sGHX7cbc)

</div>

---

## Abstract

We present an X-band radar system for Non-Line-of-Sight (NLOS) imaging that detects objects hidden from direct view by analyzing indirect signal reflections off relay surfaces. Unlike optical NLOS methods, long-wavelength X-band radar converts diffuse wall reflections into predominantly specular ones, enabling large-scale hidden-scene perception. Under equivalent transmit power, our system achieves **10× longer range than optical NLOS systems**, with real-world validation demonstrating accurate reconstructions at distances up to **40 meters**.

Our approach introduces a two-stage neural reconstruction pipeline: a **Swin-UNet dense prediction module** first identifies Line-of-Sight (LOS) and multi-bounce NLOS signal components, followed by a **geometry-aware ResidualReflectNet** that mirrors detections across relay walls with learned residual corrections. We validate the system through an end-to-end simulator (2,160 synthetic scenes) and 122 real-world captures across diverse indoor and outdoor environments.

---

## Installation

```bash
pip install -r requirements.txt
```

## Repository Structure

```
XBandNlos/
├── data/                          # datasets
│   ├── real_data/
│   │   ├── train/                 # real captures for training
│   │   └── test/                  # real captures for evaluation
│   └── simulated_data/
│       ├── train/                 # simulated scenes for training
│       ├── val/                   # simulated scenes for validation
│       └── test/                  # simulated scenes for evaluation
│
├── checkpoints/                   # pretrained weights
│   ├── swinunet_stage1_realdata.pth
│   ├── swinunet_stage1_simdata.pth
│   ├── reflectnet_stage2_realdata.pth
│   └── reflectnet_stage2_simdata.pth
│
├── results/                       # inference outputs
├── utils/                         # shared modules
├── infer_swin.py                  # Stage 1 inference
├── infer_nlos.py                  # Stage 2 inference
├── train_swin.py                  # Stage 1 training
├── train_nlos.py                  # Stage 2 training
└── requirements.txt
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

### Stage 1 — Detect LOS & mNLOS points (SwinUNet)

```bash
python train_swin.py --mode real \
    --label_dir   data/real_data/train \
    --train_list  data/data_splits/exp_train_files.txt \
    --val_list    data/data_splits/exp_test_files.txt \
    --output      checkpoints/swinunet_stage1_realdata.pth \
    --start_ckpt  checkpoints/swinunet_stage1_realdata.pth
```

### Stage 2 — NLOS Reconstruction (ResidualReflectNet)

```bash
python train_nlos.py --mode sim \
    --data_pattern data/simulated_data/train/points \
    --train_list   data/data_splits/train_files.txt \
    --test_list    data/data_splits/test_files.txt \
    --output       checkpoints/reflectnet_stage2_simdata.pth
```

## Citation

```bibtex
@inproceedings{du2026xbandnlos,
  title     = {X-Band Radar Non-Line-of-Sight Imaging},
  author    = {Du, Dongyu and Zhao, Mingkun and Yang, Yutong and Scheuble, Dominik and
               Huang, Xiaolong and Shao, Zijian and Bijelic, Mario and
               Sengupta, Kaushik and Heide, Felix},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026},
}
```
