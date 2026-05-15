"""
Radar coordinate axes, data loading, and RA-map utilities.

Coordinate conventions
  Real experiment : range 4.521 – 43.994 m, azimuth ±60°
  Simulation      : range 0 – 109.4 m,      azimuth ±60°
"""

import numpy as np
import torch
import torch.nn.functional as F
from skimage.feature import peak_local_max

# ---------------------------------------------------------------------------
# Coordinate axes
# ---------------------------------------------------------------------------
R_REAL = np.linspace(4.521, 43.994, 512).astype(np.float32)
A_REAL = np.deg2rad(np.linspace(-60.0, 60.0, 256)).astype(np.float32)

R_SIM  = np.linspace(0, 109.4, 512).astype(np.float32)
A_SIM  = np.deg2rad(np.linspace(-60.0, 60.0, 256)).astype(np.float32)


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def ra_to_cartesian(coordinates: np.ndarray,
                    R: np.ndarray, A: np.ndarray) -> np.ndarray:
    """(N, 2) [range_bin, azimuth_bin] → (N, 2) Cartesian xy metres."""
    r = R[coordinates[:, 0]]
    a = A[coordinates[:, 1]]
    return np.vstack((r * np.sin(-a), r * np.cos(a))).T


def xy_to_ra_bins(xy: np.ndarray,
                  R: np.ndarray = R_REAL,
                  A: np.ndarray = A_REAL):
    """(N, 2) Cartesian → (r_bins, a_bins); out-of-range points are discarded."""
    if xy.shape[0] == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    x, y = xy[:, 0], xy[:, 1]
    r_vals = np.sqrt(x ** 2 + y ** 2)
    a_vals = np.arctan2(-x, y).astype(np.float32)
    valid = ((r_vals >= R[0]) & (r_vals <= R[-1]) &
             (a_vals >= A[0]) & (a_vals <= A[-1]))
    if not valid.any():
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    r_bins = np.argmin(np.abs(R[:, None] - r_vals[valid][None, :]), axis=0)
    a_bins = np.argmin(np.abs(A[:, None] - a_vals[valid][None, :]), axis=0)
    return r_bins.astype(int), a_bins.astype(int)


# ---------------------------------------------------------------------------
# Data loading  →  (1, 1, 512, 512) tensor ready for SwinUNet
# ---------------------------------------------------------------------------

def load_ra_real(ra_path: str) -> torch.Tensor:
    """Load real RA .npz (Pxx key), normalise to dB, pad azimuth 256→512."""
    pxx = np.load(ra_path)['Pxx'].astype(np.float64)
    pxx = np.clip(pxx, 0, None)
    pxx = 10.0 * np.log10(pxx / (pxx.max() + 1e-12) + 1e-12)
    pxx = (pxx - pxx.min()) / (pxx.max() - pxx.min() + 1e-12)
    t   = torch.from_numpy(pxx).float().unsqueeze(0).unsqueeze(0)
    return F.pad(t, (128, 128, 0, 0), value=0.0)


def load_ra_sim(ra_path: str) -> torch.Tensor:
    """Load raw complex sim RA .npy, normalise to dB, pad azimuth 256→512."""
    ra   = np.load(ra_path)
    mag  = np.abs(ra).astype(np.float64)
    db   = 10.0 * np.log10(mag / (mag.max() + 1e-12) + 1e-12)
    norm = (db - db.min()) / (db.max() - db.min() + 1e-12)
    t    = torch.from_numpy(norm).float().unsqueeze(0).unsqueeze(0)
    return F.pad(t, (128, 128, 0, 0), value=0.0)


# ---------------------------------------------------------------------------
# Peak detection
# ---------------------------------------------------------------------------

def detect_peaks(pred_map: np.ndarray, conf_thres: float, min_dis: int,
                 R: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Local maxima in a 2-D heatmap → Cartesian (N, 2) array."""
    bins = peak_local_max(pred_map, min_distance=min_dis, threshold_abs=conf_thres)
    return ra_to_cartesian(bins, R, A) if bins.shape[0] > 0 \
           else np.empty((0, 2), dtype=np.float32)


# ---------------------------------------------------------------------------
# Point-cloud helpers
# ---------------------------------------------------------------------------

def to_3xN(xy: np.ndarray) -> np.ndarray:
    """(N, 2) Cartesian → (3, N) with a zero intensity row appended."""
    if xy.shape[0] == 0:
        return np.zeros((3, 0), dtype=np.float32)
    return np.vstack([xy.T.astype(np.float32),
                      np.zeros((1, xy.shape[0]), dtype=np.float32)])
