"""
Dataset classes for NLOS radar training.

MLNsResidualDataset  — Stage 2 (ResidualReflectNet) training on sim data.
ExpNlosDataset       — Stage 1 (SwinUNet) fine-tuning on real experiment data.
"""

import os

import numpy as np
import torch
from sklearn.cluster import DBSCAN
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Any, Dict, List, Tuple

from .radar import xy_to_ra_bins, R_REAL, A_REAL
from .utils import (
    find_lines_in_patch_iterative,
    get_fallback_line,
    get_polar_angles_torch,
    load_file_for_training,
    load_npz_for_training,
    reflect_point,
)

class MLNsResidualDataset(Dataset):
    """
    Pre-computes Scheme J predictions and prepares (X, Y) pairs
    where X is the context and Y is the residual error (T_gt - T_geom).
    """
    def __init__(self, files: List[str], params: Dict[str, Any]):
        self.samples = []
        self.params = params
        self.dbscan_min_samples = self.params['DBSCAN_MIN_SAMPLES']
        
        print(f"Loading and pre-processing {len(files)} files...")
        for path in tqdm(files):
            try:
                if path.endswith('.npz'):
                    L_np, G_np, T_gt_np = load_npz_for_training(path, self.dbscan_min_samples)
                elif path.endswith('.npy'):
                    L_np, G_np, T_gt_np = load_file_for_training(path, self.dbscan_min_samples)
                else:
                    raise ValueError(f"Unknown file type: {path}. Expected .npy or .npz")
                
                K = min(G_np.shape[0], T_gt_np.shape[0])
                if K == 0: continue
                G_np, T_gt_np = G_np[:K], T_gt_np[:K]
                
                L = torch.from_numpy(L_np)
                G = torch.from_numpy(G_np)
                T_gt = torch.from_numpy(T_gt_np)
                
                # --- Run Scheme J Geometry ---
                clustering = DBSCAN(eps=self.params['DBSCAN_EPS'], min_samples=self.params['DBSCAN_MIN_SAMPLES']).fit(L_np)
                dbscan_labels = clustering.labels_
                unique_dbscan_labels = set(dbscan_labels)

                final_line_params_map = {}
                final_point_to_line_map = {}
                final_line_id_counter = 0

                for dbscan_id in unique_dbscan_labels:
                    if dbscan_id == -1: continue
                    global_indices = np.where(dbscan_labels == dbscan_id)[0]
                    patch_points = L_np[global_indices]
                    local_map, _, local_pt_map = find_lines_in_patch_iterative(
                        patch_points, global_indices,
                        self.params['LOCAL_RANSAC_MIN_INLIERS'], 
                        self.params['LOCAL_RANSAC_ITERATIONS'], 
                        self.params['LOCAL_RANSAC_THRESHOLD']
                    )
                    for local_line_id, params_arr in local_map.items():
                        global_line_id = final_line_id_counter
                        final_line_params_map[global_line_id] = torch.from_numpy(params_arr).float()
                        for global_pt_idx, pt_line_id in local_pt_map.items():
                            if pt_line_id == local_line_id:
                                final_point_to_line_map[global_pt_idx] = global_line_id
                        final_line_id_counter += 1
                
                angles_L = get_polar_angles_torch(L)
                angles_G = get_polar_angles_torch(G)
                angle_diffs = torch.abs(angles_G.unsqueeze(1) - angles_L.unsqueeze(0))
                angle_diffs = torch.min(angle_diffs, 2 * np.pi - angle_diffs)
                buddy_indices = torch.argmin(angle_diffs, dim=1) # (K,)
                
                # --- Create Training Samples ---
                for i in range(K):
                    g_i = G[i]
                    t_gt_i = T_gt[i]
                    buddy_idx = buddy_indices[i].item()
                    l_buddy = L[buddy_idx]
                    
                    line_id = final_point_to_line_map.get(buddy_idx, -1)
                    
                    if line_id != -1:
                        is_noise_feature = torch.tensor([0.0])
                        S_i = final_line_params_map[line_id]
                    else:
                        is_noise_feature = torch.tensor([1.0])
                        S_i = get_fallback_line(l_buddy)
                        
                    T_geom_i = reflect_point(g_i.unsqueeze(0), S_i.unsqueeze(0)).squeeze(0)
                    Delta_T_gt_i = t_gt_i - T_geom_i
                    
                    buddy_dbscan_id = dbscan_labels[buddy_idx].item()
                    if buddy_dbscan_id == -1:
                        l_patch_full = l_buddy.unsqueeze(0)
                    else:
                        l_patch_full = L[dbscan_labels == buddy_dbscan_id]
                    
                    num_patch_points = l_patch_full.shape[0]
                    k_patch = self.params['K_NEIGHBORS_PATCH']
                    if num_patch_points < k_patch:
                        padding = l_buddy.unsqueeze(0).repeat(k_patch - num_patch_points, 1)
                        l_patch = torch.cat([l_patch_full, padding], dim=0)
                    else:
                        dists = torch.norm(l_patch_full - l_buddy.unsqueeze(0), dim=1)
                        topk_idx = torch.topk(dists, k=k_patch, largest=False).indices
                        l_patch = l_patch_full[topk_idx]
                        
                    self.samples.append({
                        "G_i": g_i,
                        "L_patch_i": l_patch,
                        "S_i": S_i,
                        "is_noise": is_noise_feature,
                        "Delta_T_gt": Delta_T_gt_i
                    })

            except Exception as e:
                print(f"Warning: Error processing file {path}: {e}")
                
        print(f"Successfully created {len(self.samples)} training samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


class ExpNlosDataset(Dataset):
    """Stage1 dataset for real experiment captures.

    Loads welch_interpolated_background_subtracted.npz and Gaussian-blurred
    GT heatmaps for LOS + mNLOS channels.
    """

    def __init__(self, captures: List[str], label_dir: str,
                 sigma: Tuple[float, float] = (1.0, 1.0)):
        self.label_dir = label_dir
        self.sigma     = sigma
        self.captures  = [
            c for c in captures
            if os.path.isfile(os.path.join(
                label_dir, c,
                'welch_interpolated_background_subtracted.npz'))
        ]

    def __len__(self) -> int:
        return len(self.captures)

    @staticmethod
    def _to_db(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 0, None)
        return 10.0 * np.log10(x / (np.max(x) + 1e-12) + 1e-12)

    @staticmethod
    def _gaussian_heatmap(gt_2d: np.ndarray, sigma: float) -> torch.Tensor:
        H, W = gt_2d.shape
        gt   = torch.from_numpy(gt_2d.astype(np.float32))
        ys, xs = torch.where(gt == 1)
        hm = torch.zeros_like(gt)
        yg, xg = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        for py, px in zip(ys, xs):
            d2 = (xg - px) ** 2 + (yg - py) ** 2
            hm = torch.maximum(hm, torch.exp(-d2 / (2 * sigma ** 2)))
        return hm

    def __getitem__(self, idx: int):
        cap  = self.captures[idx]
        ra_f = os.path.join(self.label_dir, cap,
                            'welch_interpolated_background_subtracted.npz')
        lp   = os.path.join(self.label_dir, cap, 'labeled_points')

        pxx = np.load(ra_f)['Pxx'].astype(np.float64)
        pxx = self._to_db(pxx)
        pxx = (pxx - pxx.min()) / (pxx.max() - pxx.min() + 1e-12)
        radar = torch.from_numpy(pxx).unsqueeze(0).float()
        radar = torch.nn.functional.pad(radar, (128, 128, 0, 0), value=0.0)

        gt_los   = np.zeros((512, 256), dtype=np.int32)
        gt_mnlos = np.zeros((512, 256), dtype=np.int32)
        for fname, gt_arr in [('los.npy', gt_los), ('mirrored_nlos.npy', gt_mnlos)]:
            fpath = os.path.join(lp, fname)
            if os.path.isfile(fpath):
                xy = np.load(fpath).astype(np.float32)
                if xy.ndim == 2:
                    rb, ab = xy_to_ra_bins(xy, R_REAL, A_REAL)
                    for r, a in zip(rb, ab):
                        gt_arr[r, a] = 1

        target = torch.stack([
            self._gaussian_heatmap(gt_los,   self.sigma[0]),
            self._gaussian_heatmap(gt_mnlos, self.sigma[1]),
        ], dim=0)

        return radar, target, cap