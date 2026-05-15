"""
Stage 2 Inference: NLOS reconstruction via geometry + ResidualReflectNet.

Takes L/G point clouds (output of infer_swin.py) and estimates hidden NLOS
target positions.

Pipeline per capture:
    1. DBSCAN clusters LOS points into wall segments
    2. Iterative RANSAC fits a line to each cluster
    3. Each mNLOS (G) point is reflected off its nearest wall → T_geom
    4. ResidualReflectNet predicts correction Δ → T_pred = T_geom + Δ

Run from the public/ directory:
    python infer_nlos.py \
        --input_dir  results/stage1_real \
        --output_dir results/nlos_recon

    With GT evaluation + figures:
    python infer_nlos.py \
        --input_dir  results/stage1_real \
        --label_dir  /path/to/data/Experiment/1112label_data \
        --output_dir results/nlos_recon \
        --save_figs
"""

import argparse
import glob
import os

import numpy as np
import torch
from tqdm import tqdm

from utils.model import ResidualReflectNet, predict_scheme_K
from utils.utils import calculate_set_metrics, f1_from_counts, reflect_point
from utils.viz import save_nlos_fig


def _fit_line(pts: np.ndarray) -> np.ndarray:
    """Fit a normalised line (a, b, c) to 2-D points via SVD."""
    mean = pts.mean(0)
    _, _, Vh = np.linalg.svd(pts - mean)
    a, b = Vh[-1]
    c    = -(a * mean[0] + b * mean[1])
    nrm  = np.sqrt(a ** 2 + b ** 2) + 1e-8
    return np.array([a / nrm, b / nrm, c / nrm], dtype=np.float32)


def _reflect_np(G_np: np.ndarray, abc: np.ndarray) -> np.ndarray:
    G_t = torch.from_numpy(G_np).float()
    S   = torch.from_numpy(abc).float().unsqueeze(0).expand(len(G_np), -1)
    return reflect_point(G_t, S).numpy()


def main():
    parser = argparse.ArgumentParser(description="Stage 2: NLOS reconstruction")
    parser.add_argument("--input_dir",  required=True,
                        help="L/G point .npz files (output of infer_swin.py)")
    parser.add_argument("--label_dir",  default="",
                        help="labeled_points/ tree for GT evaluation (optional)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/reflectnet_stage2_realdata.pth")
    parser.add_argument("--use_gt_wall", action="store_true",
                        help="Use labeled wall.npy instead of DBSCAN+RANSAC (upper-bound)")
    parser.add_argument("--dbscan_eps",         type=float, default=4.0)
    parser.add_argument("--dbscan_min_samples",  type=int,   default=2)
    parser.add_argument("--ransac_iters",        type=int,   default=200)
    parser.add_argument("--ransac_threshold",    type=float, default=0.3)
    parser.add_argument("--ransac_min_inliers",  type=int,   default=2)
    parser.add_argument("--wall_filter_ratio",   type=float, default=0.2,
                        help="Pre-filter: keep L points with y >= G_ymin * ratio (0=off)")
    parser.add_argument("--wall_y_margin",       type=float, default=3.0,
                        help="For predicted W key: keep only points within y_min + margin")
    parser.add_argument("--k_neighbors",         type=int,   default=16)
    parser.add_argument("--batch_size",          type=int,   default=256)
    parser.add_argument("--f1_threshold",        type=float, default=2.0,
                        help="Matching distance in metres for F1")
    parser.add_argument("--save_figs",  action="store_true")
    parser.add_argument("--fig_xlim",   type=float, nargs=2, default=[-40, 40])
    parser.add_argument("--fig_ylim",   type=float, nargs=2, default=[0, 45])
    parser.add_argument("--device",     default="")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    PARAMS = {
        "DEVICE":                   device,
        "DBSCAN_EPS":               args.dbscan_eps,
        "DBSCAN_MIN_SAMPLES":       args.dbscan_min_samples,
        "LOCAL_RANSAC_ITERATIONS":  args.ransac_iters,
        "LOCAL_RANSAC_THRESHOLD":   args.ransac_threshold,
        "LOCAL_RANSAC_MIN_INLIERS": args.ransac_min_inliers,
        "K_NEIGHBORS_PATCH":        args.k_neighbors,
        "BATCH_SIZE":               args.batch_size,
        "WALL_FILTER_RATIO":        args.wall_filter_ratio,
    }

    model = None
    if not args.use_gt_wall:
        model = ResidualReflectNet(k_neighbors=args.k_neighbors).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval()
        print(f"Model loaded: {args.checkpoint}")
    else:
        print("--use_gt_wall: upper-bound mode, no NN.")

    input_files = sorted(glob.glob(
        os.path.join(args.input_dir, "**", "*.npz"), recursive=True))
    input_files = [f for f in input_files if "figs" not in f.split(os.sep)]
    print(f"Found {len(input_files)} input files.")

    total_tp = total_fp = total_fn = 0
    total_cd = 0.0
    n_eval   = 0

    for input_path in tqdm(input_files, desc="Inference"):
        rel       = os.path.relpath(input_path, args.input_dir)
        data_name = rel.replace('.npz', '')
        npz_out   = os.path.join(args.output_dir, rel)
        fig_out   = os.path.join(args.output_dir, "figs", rel.replace('.npz', '.svg'))
        os.makedirs(os.path.dirname(npz_out), exist_ok=True)

        try:
            data = np.load(input_path)
            L_np = data['L'][:2, :].T.astype(np.float32)
            G_np = data['G'][:2, :].T.astype(np.float32)
        except Exception as e:
            print(f"  Skip (load error): {input_path} — {e}"); continue

        if G_np.shape[0] == 0:
            continue

        # Load GT if available (label_dir takes priority, then embedded T key)
        T_np = None
        if args.label_dir:
            nlos_path = os.path.join(args.label_dir, data_name,
                                     'labeled_points', 'nlos.npy')
            if os.path.isfile(nlos_path):
                T_np = np.load(nlos_path).astype(np.float32)
        if T_np is None and 'T' in data:
            raw_T = data['T']
            if raw_T.ndim == 2 and raw_T.shape[0] == 3:
                T_np = raw_T[:2, :].T.astype(np.float32)
            elif raw_T.ndim == 2:
                T_np = raw_T[:, :2].astype(np.float32)

        # Wall source: GT wall  >  predicted W key  >  DBSCAN+RANSAC on L
        wall_np = None
        if args.use_gt_wall:
            wall_path = os.path.join(args.label_dir, data_name,
                                     'labeled_points', 'wall.npy')
            if not os.path.isfile(wall_path):
                print(f"  Skip (wall GT missing): {wall_path}"); continue
            wall_np = np.load(wall_path).astype(np.float32)
        elif 'W' in data:
            W_raw = data['W']
            if W_raw.ndim == 2 and W_raw.shape[0] == 3 and W_raw.shape[1] > 0:
                wall_np = W_raw[:2, :].T.astype(np.float32)
            elif W_raw.ndim == 2 and W_raw.shape[1] >= 2:
                wall_np = W_raw[:, :2].astype(np.float32)
            if wall_np is not None and args.wall_y_margin > 0:
                y_min = wall_np[:, 1].min()
                wall_np = wall_np[wall_np[:, 1] <= y_min + args.wall_y_margin]

        use_wall = wall_np is not None and wall_np.shape[0] >= 2

        if use_wall:
            try:
                abc       = _fit_line(wall_np)
                T_pred_np = _reflect_np(G_np, abc)
                T_geom_np = T_pred_np.copy()
                L_plot    = np.vstack([L_np, wall_np])
                wi        = L_np.shape[0]
                cluster_labels  = np.full(L_np.shape[0], -1, dtype=int)
                line_params_map  = {0: abc}
                line_inliers_map = {0: np.arange(wi, wi + wall_np.shape[0])}
            except Exception as e:
                print(f"  Skip (wall error): {input_path} — {e}"); continue
        else:
            L = torch.from_numpy(L_np).to(device)
            G = torch.from_numpy(G_np).to(device)
            try:
                T_geom, T_pred, cluster_labels, line_params_map, line_inliers_map, *_ = \
                    predict_scheme_K(model, L, G, PARAMS)
            except Exception as e:
                print(f"  Skip (inference error): {input_path} — {e}"); continue
            T_pred_np = T_pred.cpu().numpy()
            T_geom_np = T_geom.cpu().numpy()
            L_plot    = L_np

        save_dict = dict(L=L_np, G=G_np, T_pred=T_pred_np, T_geom=T_geom_np)
        if T_np is not None:
            save_dict['T'] = T_np
        np.savez(npz_out, **save_dict)

        if T_np is not None:
            m = calculate_set_metrics(
                torch.from_numpy(T_pred_np), torch.from_numpy(T_np),
                args.f1_threshold)
            total_tp += m['tp']; total_fp += m['fp']
            total_fn += m['fn']; total_cd += m['cd']
            n_eval += 1

        if args.save_figs:
            save_nlos_fig(fig_out, L_np, T_np, T_pred_np, cluster_labels,
                          L_plot, line_params_map, line_inliers_map,
                          args.fig_xlim, args.fig_ylim)

    mode = "GT-wall" if args.use_gt_wall else "auto-wall+NN"
    print(f"\n{'='*50}")
    print(f"Mode      : {mode}")
    print(f"Processed : {n_eval} files with GT metrics")
    if n_eval > 0:
        p, r, f1 = f1_from_counts(total_tp, total_fp, total_fn)
        print(f"  Micro F1 : {f1:.4f}  (P: {p:.4f}, R: {r:.4f})")
        print(f"  Macro CD : {total_cd/n_eval:.4f} m")
    print(f"Results   : {args.output_dir}")
    print('=' * 50)


if __name__ == "__main__":
    main()
