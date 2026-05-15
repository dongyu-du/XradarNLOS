# model.py
"""
Defines the ResidualReflectNet model architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Set, List, Any
import numpy as np
from sklearn.cluster import DBSCAN

from .utils import (
    get_polar_angles_torch,
    find_lines_in_patch_iterative,
    get_fallback_line,
    reflect_point,
)

class ResidualReflectNet(nn.Module):
    """
    A residual network that learns to correct geometry-based predictions.
    """
    def __init__(self, k_neighbors: int, point_dim: int = 2, 
                 embed_dim: int = 64, patch_feature_dim: int = 128, state_feature_dim: int = 64):
        super().__init__()
        
        # L-Patch Encoder (Attention)
        self.patch_encoder = nn.Sequential(
            nn.Linear(point_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(),
            nn.Linear(embed_dim * 2, patch_feature_dim)
        )
        self.g_query_encoder = nn.Sequential(
            nn.Linear(point_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, patch_feature_dim)
        )
        
        # State Encoder
        state_input_dim = 2 + 3 + 1 # G_i(2) + S_i(3) + is_noise(1)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_input_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, state_feature_dim)
        )
        
        # Decoder
        decoder_input_dim = patch_feature_dim + state_feature_dim
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, point_dim) # Output Delta_T (2)
        )

    def forward(self, g_points: torch.Tensor, l_patches: torch.Tensor, 
                S_lines: torch.Tensor, is_noise: torch.Tensor) -> torch.Tensor:
        
        B, K, P_dim = l_patches.shape
        
        # 1. Encode L-Patch (Attention)
        l_patches_relative = l_patches - g_points.unsqueeze(1)
        l_patches_flat = l_patches_relative.view(B * K, P_dim)
        patch_features = self.patch_encoder(l_patches_flat)
        patch_values = patch_features.view(B, K, -1)
        
        g_query = self.g_query_encoder(g_points)
        g_query_expanded = g_query.unsqueeze(1)
        scores = torch.bmm(g_query_expanded, patch_values.transpose(1, 2))
        attention_weights = F.softmax(scores, dim=-1)
        
        patch_vec_weighted = torch.bmm(attention_weights, patch_values)
        v_patch = patch_vec_weighted.squeeze(1)
        
        # 2. Encode State
        state_input = torch.cat([g_points, S_lines, is_noise], dim=1)
        v_state = self.state_encoder(state_input)
        
        # 3. Decode
        combined_vec = torch.cat([v_patch, v_state], dim=1)
        Delta_T_pred = self.decoder(combined_vec)
        
        return Delta_T_pred
    
def predict_scheme_K(
    model: ResidualReflectNet, 
    L: torch.Tensor, 
    G: torch.Tensor,
    params: Dict[str, Any]
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Runs the full Scheme K inference.
    Returns T_geom, T_final (NN-corrected), and maps for plotting.
    """
    
    model.eval()
    L_np = L.cpu().numpy()
    G_np = G.cpu().numpy()

    # --- G-guided range prefilter (wall lives just below G_ymin) ---
    # Only L points with y > G_ymin * low_ratio are fed to DBSCAN/RANSAC.
    # The full L_np is kept for the NN patch step.
    wall_filter_ratio = params.get('WALL_FILTER_RATIO', 0.0)
    if wall_filter_ratio > 0 and G_np.shape[0] > 0:
        g_ymin = G_np[:, 1].min()
        wall_mask = L_np[:, 1] >= g_ymin * wall_filter_ratio
        L_wall = L_np[wall_mask]
        wall_orig_idx = np.where(wall_mask)[0]
    else:
        L_wall = L_np
        wall_orig_idx = np.arange(len(L_np))

    if len(L_wall) < 2:
        L_wall = L_np
        wall_orig_idx = np.arange(len(L_np))

    # --- 1. Run Scheme J geometry pass ---
    clustering = DBSCAN(eps=params['DBSCAN_EPS'], min_samples=params['DBSCAN_MIN_SAMPLES']).fit(L_wall)
    dbscan_labels_wall = clustering.labels_
    # map back to full L_np index space
    dbscan_labels = np.full(len(L_np), -1, dtype=int)
    dbscan_labels[wall_orig_idx] = dbscan_labels_wall
    unique_dbscan_labels = set(dbscan_labels_wall)

    final_line_params_map = {}   
    final_line_inliers_map = {}  
    final_point_to_line_map = {} 
    final_line_cluster_labels = np.full(L_np.shape[0], -1, dtype=int)
    final_line_id_counter = 0

    for dbscan_id in unique_dbscan_labels:
        if dbscan_id == -1: continue
        global_indices = np.where(dbscan_labels == dbscan_id)[0]
        patch_points = L_np[global_indices]
        local_map, local_inliers, local_pt_map = find_lines_in_patch_iterative(
            patch_points, global_indices,
            params['LOCAL_RANSAC_MIN_INLIERS'], 
            params['LOCAL_RANSAC_ITERATIONS'], 
            params['LOCAL_RANSAC_THRESHOLD']
        )
        for local_line_id, params_arr in local_map.items():
            global_line_id = final_line_id_counter
            final_line_params_map[global_line_id] = params_arr 
            final_line_inliers_map[global_line_id] = local_inliers[local_line_id]
            for global_pt_idx, pt_line_id in local_pt_map.items():
                if pt_line_id == local_line_id:
                    final_point_to_line_map[global_pt_idx] = global_line_id
                    final_line_cluster_labels[global_pt_idx] = global_line_id
            final_line_id_counter += 1
    
    angles_L = get_polar_angles_torch(L)
    angles_G = get_polar_angles_torch(G)
    angle_diffs = torch.abs(angles_G.unsqueeze(1) - angles_L.unsqueeze(0))
    angle_diffs = torch.min(angle_diffs, 2 * np.pi - angle_diffs)
    buddy_indices = torch.argmin(angle_diffs, dim=1)
    
    # --- 2. Prepare batches for NN ---
    K = G.shape[0]
    batch_g_i, batch_l_patch_i, batch_S_i, batch_is_noise, batch_T_geom = [], [], [], [], []
    
    k_patch = params['K_NEIGHBORS_PATCH']
    device = params['DEVICE']

    for i in range(K):
        g_i = G[i]
        buddy_idx = buddy_indices[i].item()
        l_buddy = L[buddy_idx]
        
        line_id = final_point_to_line_map.get(buddy_idx, -1)
        
        if line_id != -1:
            is_noise_feature = torch.tensor([0.0], device=device)
            S_i_np = final_line_params_map[line_id]
            S_i = torch.from_numpy(S_i_np).to(device).float()
        else:
            is_noise_feature = torch.tensor([1.0], device=device)
            S_i = get_fallback_line(l_buddy).to(device)
            
        T_geom_i = reflect_point(g_i.unsqueeze(0), S_i.unsqueeze(0)).squeeze(0)
        
        buddy_dbscan_id = dbscan_labels[buddy_idx].item()
        if buddy_dbscan_id == -1:
            l_patch_full = l_buddy.unsqueeze(0)
        else:
            l_patch_full = L[dbscan_labels == buddy_dbscan_id]
        
        num_patch_points = l_patch_full.shape[0]
        if num_patch_points < k_patch:
            padding = l_buddy.unsqueeze(0).repeat(k_patch - num_patch_points, 1)
            l_patch = torch.cat([l_patch_full, padding], dim=0)
        else:
            dists = torch.norm(l_patch_full - l_buddy.unsqueeze(0), dim=1)
            topk_idx = torch.topk(dists, k=k_patch, largest=False).indices
            l_patch = l_patch_full[topk_idx]
            
        batch_g_i.append(g_i.unsqueeze(0))
        batch_l_patch_i.append(l_patch.unsqueeze(0))
        batch_S_i.append(S_i.unsqueeze(0))
        batch_is_noise.append(is_noise_feature.unsqueeze(0))
        batch_T_geom.append(T_geom_i.unsqueeze(0))

    g_tensor = torch.cat(batch_g_i, dim=0)
    l_patch_tensor = torch.cat(batch_l_patch_i, dim=0)
    S_tensor = torch.cat(batch_S_i, dim=0)
    is_noise_tensor = torch.cat(batch_is_noise, dim=0)
    T_geom_tensor = torch.cat(batch_T_geom, dim=0)
    
    # --- 3. Run NN Inference ---
    Delta_T_pred_tensor = torch.zeros_like(T_geom_tensor)
    with torch.no_grad():
        for i in range(0, K, params['BATCH_SIZE']):
            Delta_T_pred = model(
                g_tensor[i : i + params['BATCH_SIZE']],
                l_patch_tensor[i : i + params['BATCH_SIZE']],
                S_tensor[i : i + params['BATCH_SIZE']],
                is_noise_tensor[i : i + params['BATCH_SIZE']]
            )
            Delta_T_pred_tensor[i : i + params['BATCH_SIZE']] = Delta_T_pred
            
    # --- 4. Calculate Final Prediction ---
    T_final_tensor = T_geom_tensor + Delta_T_pred_tensor
    
    return (T_geom_tensor, T_final_tensor, 
            final_line_cluster_labels,
            final_line_params_map, final_line_inliers_map,buddy_indices,S_tensor)