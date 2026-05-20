"""
Stage 1 inference for ALL simulation data (train + val + test combined).

Scans every split directory (train/RA, val/RA, test/RA), runs SwinUNet, and
writes results organised by scene instead of by split:

    output_dir/
        scene1/ *.npz
        scene2/ *.npz
        scene3/ *.npz
        figs/scene1/ *.svg
        figs/scene2/ *.svg
        figs/scene3/ *.svg

Each .npz has keys: L  (3, N)  LOS points
                    G  (3, N)  mNLOS points
                    T  (3, M)  NLOS ground-truth

Run from the public/ directory:

    python infer_swin_simall.py \
        --sim_root  data/simulated_data \
        --output_dir /scratch/.../results/sim_all/stage1 \
        --splits train val test \
        --save_figs
"""

import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from utils.swin import SwinUnet
from utils.radar import R_SIM, A_SIM, load_ra_sim, detect_peaks, to_3xN
from utils.viz import save_sim_fig


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 sim inference — output by scene",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sim_root",   required=True,
                        help="Root of simulated_data/ (contains train/, val/, test/)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--splits",     nargs="+", default=["train", "val", "test"],
                        help="Which split directories to process")
    parser.add_argument("--checkpoint", default="checkpoints/swinunet_stage1_simdata.pth")
    parser.add_argument("--conf_thres", type=float, default=0.3)
    parser.add_argument("--min_dis",    type=int,   default=3)
    parser.add_argument("--num_classes",type=int,   default=2)
    parser.add_argument("--save_figs",  action="store_true")
    parser.add_argument("--device",     default="")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SwinUnet(img_size=512, num_classes=args.num_classes).to(device)
    missing, _ = model.load_state_dict(
        torch.load(args.checkpoint, map_location=device), strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    model.eval()
    print(f"Checkpoint: {args.checkpoint}")

    # Collect all (ra_path, gt_path, scene, file_id) tuples across all splits
    jobs = []
    for split in args.splits:
        ra_root = os.path.join(args.sim_root, split, "RA")
        gt_root = os.path.join(args.sim_root, split, "points")
        if not os.path.isdir(ra_root):
            print(f"  Skipping split '{split}': RA dir not found ({ra_root})")
            continue
        for scene in sorted(os.listdir(ra_root)):
            scene_ra = os.path.join(ra_root, scene)
            if not os.path.isdir(scene_ra):
                continue
            for fname in sorted(os.listdir(scene_ra)):
                if not fname.endswith(".npy"):
                    continue
                file_id = fname[:-4]  # strip .npy
                ra_path = os.path.join(scene_ra, fname)
                gt_path = os.path.join(gt_root, scene, file_id + ".npz")
                jobs.append((ra_path, gt_path, scene, file_id))

    print(f"Total files to process: {len(jobs)}")

    n_ok = 0
    for ra_path, gt_path, scene, file_id in tqdm(jobs, desc="Stage1"):
        out_path = os.path.join(args.output_dir, scene, file_id + ".npz")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if not os.path.isfile(gt_path):
            tqdm.write(f"  Skip (GT missing): {gt_path}")
            continue
        try:
            radar = load_ra_sim(ra_path).to(device)
        except Exception as e:
            tqdm.write(f"  Skip (load error) {scene}/{file_id}: {e}")
            continue

        with torch.no_grad():
            pred = model(radar)[..., 128:128 + 256]
        pred = pred.squeeze(0).sigmoid().cpu().numpy()

        los_xy = detect_peaks(pred[0], args.conf_thres, args.min_dis, R_SIM, A_SIM)
        mn_xy  = detect_peaks(pred[1], args.conf_thres, args.min_dis, R_SIM, A_SIM)

        gt_data = np.load(gt_path)
        np.savez(out_path,
                 L=to_3xN(los_xy),
                 G=to_3xN(mn_xy),
                 T=gt_data['T'])
        n_ok += 1

        if args.save_figs:
            fig_path = os.path.join(args.output_dir, "figs",
                                    scene, file_id + ".svg")
            save_sim_fig(fig_path, gt_data['L'][:2], gt_data['G'][:2],
                         los_xy, mn_xy)

    print(f"\nDone. Processed {n_ok}/{len(jobs)} files → {args.output_dir}")


if __name__ == "__main__":
    main()
