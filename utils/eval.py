"""
Stage 1 evaluation for real experiment data.
"""

import os

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.feature import peak_local_max
from tqdm import tqdm

from .radar import R_REAL, A_REAL, ra_to_cartesian, load_ra_real


def _f1_counts(gt_xy: np.ndarray, pred_xy: np.ndarray,
               thresh: float):
    """Bipartite-matching F1 counts for a single frame."""
    if pred_xy.shape[0] == 0:
        return 0, 0, gt_xy.shape[0]
    if gt_xy.shape[0] == 0:
        return 0, pred_xy.shape[0], 0
    D      = cdist(gt_xy, pred_xy)
    gi, pi = linear_sum_assignment(D)
    tp     = int((D[gi, pi] < thresh).sum())
    return tp, pred_xy.shape[0] - tp, gt_xy.shape[0] - tp


def evaluate_swin(model, captures, label_dir: str, device,
                  conf_thres: float = 0.15, min_dis: int = 3,
                  match_dist: float = 2.0) -> float:
    """Evaluate Stage1 SwinUNet on real experiment captures.

    Returns macro-F1 averaged over LOS and mNLOS channels.
    """
    model.eval()
    tp_l = fp_l = fn_l = 0
    tp_m = fp_m = fn_m = 0

    for cap in tqdm(captures, desc="  eval", leave=False):
        ra_f = os.path.join(label_dir, cap,
                            'welch_interpolated_background_subtracted.npz')
        if not os.path.isfile(ra_f):
            continue

        radar = load_ra_real(ra_f)
        with torch.no_grad():
            pred = model(radar.to(device))
            pred = pred[..., 128:128 + 256].squeeze(0).sigmoid().cpu().numpy()

        lp = os.path.join(label_dir, cap, 'labeled_points')
        for ch_idx, fname in [(0, 'los.npy'), (1, 'mirrored_nlos.npy')]:
            coords  = peak_local_max(pred[ch_idx], min_distance=min_dis,
                                     threshold_abs=conf_thres)
            xy_pred = ra_to_cartesian(coords, R_REAL, A_REAL) if coords.shape[0] > 0 \
                      else np.empty((0, 2), np.float32)
            fpath   = os.path.join(lp, fname)
            xy_gt   = np.load(fpath).astype(np.float32) \
                      if os.path.isfile(fpath) else np.empty((0, 2), np.float32)
            if xy_gt.ndim == 1:
                xy_gt = np.empty((0, 2), np.float32)
            tp, fp, fn = _f1_counts(xy_gt, xy_pred, match_dist)
            if ch_idx == 0:
                tp_l += tp; fp_l += fp; fn_l += fn
            else:
                tp_m += tp; fp_m += fp; fn_m += fn

    def _f1(tp, fp, fn):
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        return 2 * p * r / (p + r + 1e-9)

    f1_los = _f1(tp_l, fp_l, fn_l)
    f1_mn  = _f1(tp_m, fp_m, fn_m)
    macro  = (f1_los + f1_mn) / 2.0
    print(f"    LOS F1={f1_los:.4f}  mNLOS F1={f1_mn:.4f}  Macro-F1={macro:.4f}")
    return macro
