# utils.py
"""
Contains helper functions for data loading, geometric operations,
and evaluation metrics.
"""

import torch
import numpy as np
from sklearn.cluster import DBSCAN
import random
from typing import Tuple, Dict, Any

# --- Data Loading ---

def to_xy_np(points_3d: np.ndarray) -> np.ndarray:
    """Extract (X, Y) coordinates from (N, 3) array."""
    if points_3d.shape[-1] != 3:
        raise ValueError(f"Expected (N, 3), got {points_3d.shape}")
    return points_3d[:, :2].copy()

def load_file_for_training(path: str, dbscan_min_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate a single .npy file."""
    A = np.load(path)
    if A.ndim != 3 or A.shape[0] != 3 or A.shape[2] != 3:
        raise ValueError(f"Expected (3, N, 3), got {A.shape} in {path}")
    L = to_xy_np(A[:, :, 0].T.astype(np.float32))
    G = to_xy_np(A[:, :, 1].T.astype(np.float32))
    T = to_xy_np(A[:, :, 2].T.astype(np.float32))
    if L.shape[0] < dbscan_min_samples or G.shape[0] == 0 or T.shape[0] == 0:
        raise ValueError(f"Not enough points in {path}")
    return L, G, T

def load_npz_for_training(path: str, dbscan_min_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and validate a single .npz file.
    Assumes 'L', 'G', 'T' keys with shape (3, N).
    """
    try:
        data = np.load(path)
    except Exception as e:
        raise IOError(f"Could not read .npz file at {path}: {e}")
    if 'L' not in data or 'G' not in data or 'T' not in data:
        raise KeyError(f"File {path} is missing one of the required keys: 'L', 'G', or 'T'.")
    L_raw = data['L'].astype(np.float32); G_raw = data['G'].astype(np.float32); T_raw = data['T'].astype(np.float32)
    if L_raw.shape[0] != 3 or G_raw.shape[0] != 3 or T_raw.shape[0] != 3:
        raise ValueError(f"Expected arrays with shape (3, N), but got L:{L_raw.shape}, G:{G_raw.shape}, T:{T_raw.shape} in {path}")
    L_transposed = L_raw.T; G_transposed = G_raw.T; T_transposed = T_raw.T
    L_np = to_xy_np(L_transposed); G_np = to_xy_np(G_transposed); T_np = to_xy_np(T_transposed)
    if L_np.shape[0] < dbscan_min_samples or G_np.shape[0] == 0 or T_np.shape[0] == 0:
        raise ValueError(f"Not enough points in {path}: L={L_np.shape[0]}, G={G_np.shape[0]}, T={T_np.shape[0]}")
    return L_np, G_np, T_np

# --- Geometric Operations ---

def get_polar_angles_torch(points: torch.Tensor) -> torch.Tensor:
    """Calculate polar angles (theta) for (N, 2) points."""
    return torch.atan2(points[:, 1], points[:, 0])

def reflect_point(g_points: torch.Tensor, line_params: torch.Tensor) -> torch.Tensor:
    """Reflect (K, 2) G-points off (K, 3) line parameters."""
    a_b = line_params[:, 0:2]
    c = line_params[:, 2:3]
    d = torch.sum(a_b * g_points, dim=1, keepdim=True) + c
    return g_points - 2 * d * a_b

def find_lines_in_patch_iterative(
    patch_points: np.ndarray, 
    global_indices: np.ndarray,
    min_inliers: int,
    iterations: int, 
    threshold: float
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, int]]:
    """Runs iterative RANSAC within a single DBSCAN cluster."""
    
    line_params_map = {}
    line_inliers_map = {}
    point_to_line_map = {}
    current_line_id = 0
    remaining_local_indices = list(range(patch_points.shape[0]))
    
    while len(remaining_local_indices) >= min_inliers:
        current_points = patch_points[remaining_local_indices]
        best_line_params = None
        best_inlier_mask_local = np.zeros(len(remaining_local_indices), dtype=bool)
        max_inliers = 0
        if current_points.shape[0] < 2: break
        for _ in range(iterations):
            try:
                sample_indices = random.sample(range(len(remaining_local_indices)), 2)
            except ValueError:
                break
            p1, p2 = current_points[sample_indices]
            if np.linalg.norm(p1 - p2) < 1e-6: continue
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            a, b = -dy, dx
            c = -(a * p1[0] + b * p1[1])
            norm_factor = np.sqrt(a**2 + b**2) + 1e-8
            a, b, c = a / norm_factor, b / norm_factor, c / norm_factor
            distances = np.abs(a * current_points[:, 0] + b * current_points[:, 1] + c)
            inlier_mask_local = distances < threshold
            num_inliers = np.sum(inlier_mask_local)
            if num_inliers > max_inliers:
                max_inliers = num_inliers
                best_line_params = np.array([a, b, c])
                best_inlier_mask_local = inlier_mask_local
        
        if max_inliers >= min_inliers and best_line_params is not None:
            inlier_points = current_points[best_inlier_mask_local]
            mean = np.mean(inlier_points, axis=0)
            centered_points = inlier_points - mean
            U, S, Vh = np.linalg.svd(centered_points)
            a, b = Vh[-1]
            c = -(a * mean[0] + b * mean[1])
            best_line_params = np.array([a, b, c])
            line_params_map[current_line_id] = best_line_params
            new_remaining_local_indices = []
            found_global_inlier_indices = []
            for i, is_inlier in enumerate(best_inlier_mask_local):
                local_idx = remaining_local_indices[i]
                global_idx = global_indices[local_idx]
                if is_inlier:
                    point_to_line_map[global_idx] = current_line_id
                    found_global_inlier_indices.append(global_idx)
                else:
                    new_remaining_local_indices.append(local_idx)
            line_inliers_map[current_line_id] = np.array(found_global_inlier_indices)
            remaining_local_indices = new_remaining_local_indices
            current_line_id += 1
        else:
            break
    return line_params_map, line_inliers_map, point_to_line_map

def get_fallback_line(l_buddy: torch.Tensor) -> torch.Tensor:
    """Fallback line: normal is vector from origin to anchor."""
    normal_vec = l_buddy / (torch.norm(l_buddy) + 1e-8)
    a, b = normal_vec[0], normal_vec[1]
    c = -(a * l_buddy[0] + b * l_buddy[1])
    return torch.tensor([a, b, c], device=l_buddy.device, dtype=torch.float32)

# --- Metrics Functions ---

def calculate_set_metrics(
    T_pred: torch.Tensor, 
    T_gt: torch.Tensor, 
    f1_threshold: float
) -> Dict[str, Any]:
    """
    Calculates UNORDERED metrics (F1 components and Chamfer Distance)
    for a single frame.
    
    Args:
        T_pred (torch.Tensor): Prediction tensor (K, 2)
        T_gt (torch.Tensor): Ground truth tensor (P, 2)
        f1_threshold (float): Matching distance in meters for F1 score.
        
    Returns:
        Dict containing 'tp', 'fp', 'fn', and 'cd'.
    """
    
    # Handle empty predictions or ground truth
    if T_pred.shape[0] == 0 or T_gt.shape[0] == 0:
        return {
            "tp": 0,
            "fp": T_pred.shape[0], # All predictions are False Positives
            "fn": T_gt.shape[0], # All ground truths are False Negatives
            "cd": float('inf')
        }

    T_pred_safe = T_pred.float()
    T_gt_safe = T_gt.float()
    
    # Calculate pairwise distance matrix (K_pred, P_gt)
    dists = torch.cdist(T_pred_safe, T_gt_safe)
    
    # 1. Calculate F1 Score Components (Micro-average)
    #    (Based on your previous eval function's logic)
    
    # For each GT point, find the distance to the *closest* Pred point
    min_dists_gt_to_pred, _ = torch.min(dists, dim=0)
    
    # For each Pred point, find the distance to the *closest* GT point
    min_dists_pred_to_gt, _ = torch.min(dists, dim=1)

    # True Positives: GT points that *have* a Pred point within the threshold
    tp = torch.sum(min_dists_gt_to_pred < f1_threshold).item()
    
    # False Negatives: GT points that *do not* have a Pred point within the threshold
    fn = torch.sum(min_dists_gt_to_pred >= f1_threshold).item()
    
    # False Positives: Pred points that *do not* have a GT point within the threshold
    fp = torch.sum(min_dists_pred_to_gt >= f1_threshold).item()
    
    # 2. Calculate Chamfer Distance (Macro-average)
    #    (Note: This is the two-sided CD)
    cd = torch.mean(min_dists_pred_to_gt) + torch.mean(min_dists_gt_to_pred)
    
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "cd": cd.item()
    }
    
def f1_from_counts(tp, fp, fn):
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    return precision, recall, f1