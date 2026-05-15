"""
Stage 2 Training: ResidualReflectNet on simulation data.

The network learns to correct geometry-based NLOS predictions by predicting
a residual Δ such that T_pred = T_geom + Δ, where T_geom is obtained by
reflecting each mNLOS (G) point off an estimated wall (DBSCAN + RANSAC on L).

Run from the public/ directory:
    python train_nlos.py --mode sim \
        --data_pattern results/SwinUNet \
        --train_list   data/data_splits/train_files.txt \
        --test_list    data/data_splits/test_files.txt  \
        --output       checkpoints/reflectnet_stage2_simdata.pth
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import MLNsResidualDataset
from utils.model import ResidualReflectNet, predict_scheme_K
from utils.utils import f1_from_counts, calculate_set_metrics, load_npz_for_training


def _train(model, dataloader, params):
    optimizer = optim.Adam(model.parameters(), lr=params['LEARNING_RATE'])
    criterion = nn.L1Loss()
    print(f"Training on {params['DEVICE']} for {params['EPOCHS']} epochs (L1 loss)")

    for epoch in range(params['EPOCHS']):
        model.train()
        total_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{params['EPOCHS']}", leave=False):
            pred = model(batch["G_i"].to(params['DEVICE']),
                         batch["L_patch_i"].to(params['DEVICE']),
                         batch["S_i"].to(params['DEVICE']),
                         batch["is_noise"].to(params['DEVICE']))
            loss = criterion(pred, batch["Delta_T_gt"].to(params['DEVICE']))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == params['EPOCHS'] - 1:
            print(f"  Epoch {epoch+1}/{params['EPOCHS']}  avg L1: "
                  f"{total_loss / len(dataloader):.4f} m")

    torch.save(model.state_dict(), params['MODEL_SAVE_PATH'])
    print(f"Saved → {params['MODEL_SAVE_PATH']}")


def _evaluate(model, test_files, params):
    model.eval()
    tp_g = fp_g = fn_g = 0
    tp_n = fp_n = fn_n = 0
    cd_g = cd_n = 0.0
    n = 0

    for path in tqdm(test_files, desc="Evaluating"):
        try:
            L_np, G_np, T_np = load_npz_for_training(path, params['DBSCAN_MIN_SAMPLES'])
            L  = torch.from_numpy(L_np).to(params['DEVICE'])
            G  = torch.from_numpy(G_np).to(params['DEVICE'])
            Tg = torch.from_numpy(T_np).to(params['DEVICE'])
            if G.shape[0] == 0:
                continue
            T_geom, T_final, *_ = predict_scheme_K(model, L, G, params)
            mg = calculate_set_metrics(T_geom,  Tg, params['F1_THRESHOLD_METERS'])
            mn = calculate_set_metrics(T_final, Tg, params['F1_THRESHOLD_METERS'])
            tp_g += mg['tp']; fp_g += mg['fp']; fn_g += mg['fn']; cd_g += mg['cd']
            tp_n += mn['tp']; fp_n += mn['fp']; fn_n += mn['fn']; cd_n += mn['cd']
            n += 1
        except Exception as e:
            print(f"  Warning {path}: {e}")

    if n == 0:
        print("No test files evaluated."); return
    p, r, f1 = f1_from_counts(tp_g, fp_g, fn_g)
    print(f"\nGeometry only  — F1:{f1:.4f} (P:{p:.4f} R:{r:.4f})  CD:{cd_g/n:.4f} m")
    p, r, f1 = f1_from_counts(tp_n, fp_n, fn_n)
    print(f"NN corrected   — F1:{f1:.4f} (P:{p:.4f} R:{r:.4f})  CD:{cd_n/n:.4f} m")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2 training: ResidualReflectNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode",         default="sim", choices=["sim"])
    parser.add_argument("--data_pattern", required=True,
                        help="Directory containing Stage1 output .npz files")
    parser.add_argument("--train_list",   required=True)
    parser.add_argument("--test_list",    required=True)
    parser.add_argument("--output",       required=True, help="Save path for trained model (.pth)")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training; load --output checkpoint and evaluate only")
    parser.add_argument("--epochs",            type=int,   default=100)
    parser.add_argument("--batch_size",        type=int,   default=128)
    parser.add_argument("--lr",                type=float, default=1e-4)
    parser.add_argument("--dbscan_eps",        type=float, default=10.0)
    parser.add_argument("--dbscan_min_samples", type=int,  default=3)
    parser.add_argument("--ransac_iters",      type=int,   default=100)
    parser.add_argument("--ransac_threshold",  type=float, default=0.5)
    parser.add_argument("--ransac_min_inliers", type=int,  default=3)
    parser.add_argument("--k_neighbors",       type=int,   default=16)
    parser.add_argument("--f1_threshold",      type=float, default=1.0,
                        help="Matching distance in metres for F1")
    parser.add_argument("--device",            default="")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Mode: {args.mode}  |  Device: {device}")

    PARAMS = {
        "DEVICE":                   device,
        "MODEL_SAVE_PATH":          args.output,
        "DBSCAN_EPS":               args.dbscan_eps,
        "DBSCAN_MIN_SAMPLES":       args.dbscan_min_samples,
        "LOCAL_RANSAC_ITERATIONS":  args.ransac_iters,
        "LOCAL_RANSAC_THRESHOLD":   args.ransac_threshold,
        "LOCAL_RANSAC_MIN_INLIERS": args.ransac_min_inliers,
        "K_NEIGHBORS_PATCH":        args.k_neighbors,
        "BATCH_SIZE":               256,
        "EPOCHS":                   args.epochs,
        "LEARNING_RATE":            args.lr,
        "F1_THRESHOLD_METERS":      args.f1_threshold,
    }

    def load_list(list_path):
        paths = np.loadtxt(list_path, dtype=str)
        return [os.path.join(args.data_pattern, f.replace(".npy", ".npz")) for f in paths]

    test_files = load_list(args.test_list)
    print(f"Test files: {len(test_files)}")

    model = ResidualReflectNet(k_neighbors=args.k_neighbors).to(device)

    if not args.skip_training:
        train_files = load_list(args.train_list)
        print(f"Train files: {len(train_files)}")
        dataset = MLNsResidualDataset(train_files, PARAMS)
        loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=os.cpu_count() // 2 or 4, pin_memory=True)
        _train(model, loader, PARAMS)
    else:
        print("--skip_training: loading checkpoint for evaluation only.")

    if not os.path.isfile(args.output):
        print(f"Error: checkpoint not found at {args.output}"); return
    model.load_state_dict(torch.load(args.output, map_location=device))
    print(f"Loaded: {args.output}")
    _evaluate(model, test_files, PARAMS)


if __name__ == "__main__":
    main()
