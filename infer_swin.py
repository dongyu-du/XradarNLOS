"""
Stage 1 Inference: SwinUNet on Range-Azimuth spectrograms.

Use --mode real  for real X-band experiment data.
Use --mode sim   for simulation data.

Run from the public/ directory:

  Real:
    python infer_swin.py --mode real \
        --input_dir  /path/to/data/Experiment/1112label_data \
        --output_dir results/stage1_real

  Sim:
    python infer_swin.py --mode sim \
        --input_dir  /path/to/hardware_config/RAs \
        --gt_dir     data/simulated_data/test/points \
        --file_list  data/data_splits/test_files.txt \
        --output_dir results/stage1_sim

Real  — discovers captures via glob (scene*/capture*/welch_…npz).
Sim   — reads a text file list of relative .npy paths.
Output .npz per capture: keys L, G (and T for sim).
"""

import argparse
import glob
import os

import numpy as np
import torch
from tqdm import tqdm

from utils.swin import SwinUnet
from utils.radar import (R_REAL, A_REAL, R_SIM, A_SIM,
                          load_ra_real, load_ra_sim,
                          detect_peaks, to_3xN)
from utils.viz import save_sim_fig


def run_real(args, model, device):
    R, A = R_REAL, A_REAL
    ra_files = sorted(glob.glob(
        os.path.join(args.input_dir, '*', '*',
                     'welch_interpolated_background_subtracted.npz')))
    print(f"Found {len(ra_files)} captures.")

    n_ok = 0
    for ra_path in tqdm(ra_files, desc="Inference"):
        cap_dir    = os.path.dirname(ra_path)
        cap_name   = os.path.basename(cap_dir)
        scene_name = os.path.basename(os.path.dirname(cap_dir))
        out_path   = os.path.join(args.output_dir, scene_name, f"{cap_name}.npz")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        try:
            radar = load_ra_real(ra_path).to(device)
        except Exception as e:
            print(f"  Skip {scene_name}/{cap_name}: {e}"); continue

        with torch.no_grad():
            pred = model(radar)[..., 128:128 + 256]
        pred = pred.squeeze(0).sigmoid().cpu().numpy()

        save_dict = dict(L=to_3xN(detect_peaks(pred[0], args.conf_thres, args.min_dis, R, A)),
                         G=to_3xN(detect_peaks(pred[1], args.conf_thres, args.min_dis, R, A)))
        if args.num_classes >= 3:
            save_dict['W'] = to_3xN(detect_peaks(pred[2], args.conf_thres, args.min_dis, R, A))
        np.savez(out_path, **save_dict)
        n_ok += 1

    print(f"\nDone. Processed {n_ok}/{len(ra_files)} captures → {args.output_dir}")


def run_sim(args, model, device):
    R, A = R_SIM, A_SIM

    scene_remap = {}
    if args.scene_map:
        for pair in args.scene_map.split(','):
            k, v = pair.strip().split('=')
            scene_remap[k.strip()] = v.strip()

    file_list = np.loadtxt(args.file_list, dtype=str)
    print(f"Files to process: {len(file_list)}")

    n_ok = 0
    for rel_path in tqdm(file_list, desc="Inference"):
        if scene_remap:
            parts = rel_path.replace("\\", "/").split("/")
            parts[0] = scene_remap.get(parts[0], parts[0])
            ra_rel = "/".join(parts)
        else:
            ra_rel = rel_path
        ra_path  = os.path.join(args.input_dir, ra_rel)
        gt_path  = os.path.join(args.gt_dir, rel_path.replace(".npy", ".npz"))
        out_path = os.path.join(args.output_dir, rel_path.replace(".npy", ".npz"))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if not os.path.isfile(ra_path):
            print(f"  Skip (RA missing): {ra_path}"); continue
        if not os.path.isfile(gt_path):
            print(f"  Skip (GT missing): {gt_path}"); continue
        try:
            radar = load_ra_sim(ra_path).to(device)
        except Exception as e:
            print(f"  Skip {rel_path}: {e}"); continue

        with torch.no_grad():
            pred = model(radar)[..., 128:128 + 256]
        pred = pred.squeeze(0).sigmoid().cpu().numpy()

        los_xy = detect_peaks(pred[0], args.conf_thres, args.min_dis, R, A)
        mn_xy  = detect_peaks(pred[1], args.conf_thres, args.min_dis, R, A)

        gt_data = np.load(gt_path)
        np.savez(out_path, L=to_3xN(los_xy), G=to_3xN(mn_xy), T=gt_data['T'])
        n_ok += 1

        if args.save_figs:
            fig_path = os.path.join(args.output_dir, "figs",
                                    rel_path.replace(".npy", ".svg"))
            save_sim_fig(fig_path, gt_data['L'][:2], gt_data['G'][:2],
                         los_xy, mn_xy)

    print(f"\nDone. Processed {n_ok}/{len(file_list)} files → {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 inference: SwinUNet on RA maps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode",        required=True, choices=["real", "sim"])
    parser.add_argument("--input_dir",   required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--checkpoint",  default="",
                        help="SwinUNet .pth; defaults to checkpoints/swinunet_stage1_{mode}data.pth")
    parser.add_argument("--conf_thres",  type=float, default=0.0,
                        help="Peak confidence threshold (default: 0.15 real / 0.3 sim)")
    parser.add_argument("--min_dis",     type=int,   default=3)
    parser.add_argument("--num_classes", type=int,   default=2,
                        help="2=LOS+mNLOS, 3=LOS+mNLOS+wall")
    parser.add_argument("--device",      default="")
    # Sim-only
    parser.add_argument("--gt_dir",    default="", help="[sim] GT .npz root dir")
    parser.add_argument("--file_list", default="", help="[sim] text file of relative .npy paths")
    parser.add_argument("--scene_map", default="", help="[sim] e.g. 'scene1=30,scene2=31'")
    parser.add_argument("--save_figs", action="store_true", help="[sim] save SVG figures")
    args = parser.parse_args()

    if not args.checkpoint:
        args.checkpoint = f"checkpoints/swinunet_stage1_{args.mode}data.pth"
    if args.conf_thres == 0.0:
        args.conf_thres = 0.15 if args.mode == "real" else 0.3

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Mode: {args.mode}  |  Device: {device}")

    model = SwinUnet(img_size=512, num_classes=args.num_classes).to(device)
    missing, _ = model.load_state_dict(
        torch.load(args.checkpoint, map_location=device), strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    model.eval()
    print(f"Checkpoint: {args.checkpoint}")

    if args.mode == "real":
        run_real(args, model, device)
    else:
        if not args.gt_dir or not args.file_list:
            parser.error("--gt_dir and --file_list are required for --mode sim")
        run_sim(args, model, device)


if __name__ == "__main__":
    main()
