"""
Visualization helpers for radar NLOS inference results.
All functions save SVG files and do not display anything interactively.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def save_sim_fig(fig_path: str,
                 gt_L: np.ndarray, gt_G: np.ndarray,
                 los_xy: np.ndarray, mn_xy: np.ndarray) -> None:
    """SVG scatter plot for one simulation Stage1 inference result.

    Args:
        fig_path: output .svg path (parent dirs created automatically).
        gt_L:     (2, N) GT LOS points.
        gt_G:     (2, M) GT mNLOS points.
        los_xy:   (K, 2) predicted LOS points.
        mn_xy:    (K, 2) predicted mNLOS points.
    """
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.add_patch(patches.Wedge((0, 0), 100, 30, 150, facecolor='#FBE5D6'))
    ax.scatter(gt_L[0], gt_L[1], c='#a6cee3', s=40, label='LOS GT')
    ax.scatter(gt_G[0], gt_G[1], c='#fdbf6f', s=40, label='mNLOS GT')
    if los_xy.shape[0] > 0:
        ax.scatter(los_xy[:, 0], los_xy[:, 1],
                   c='#1f78b4', s=35, marker='+', label='LOS Est.')
    if mn_xy.shape[0] > 0:
        ax.scatter(mn_xy[:, 0], mn_xy[:, 1],
                   c='orange', s=35, marker='+', label='mNLOS Est.')
    ax.set_xlim(-90, 90)
    ax.set_ylim(0, 100.5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(fig_path, format='svg', bbox_inches='tight', transparent=True)
    plt.close(fig)


def save_nlos_fig(fig_out: str,
                  L_np: np.ndarray,
                  T_np,          # (K, 2) GT or None
                  T_pred_np: np.ndarray,
                  cluster_labels: np.ndarray,
                  L_plot: np.ndarray,
                  line_params_map: dict,
                  line_inliers_map: dict,
                  xlim=(- 40, 40),
                  ylim=(0, 45)) -> None:
    """SVG scatter + wall plot for one Stage2 NLOS reconstruction result."""
    os.makedirs(os.path.dirname(fig_out), exist_ok=True)
    COLOR_LOS  = '#574240'
    COLOR_GT   = '#F3A8A5'
    COLOR_PRED = '#D76364'
    COLOR_WALL = 'orange'

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    noise_mask = (cluster_labels == -1)
    ax.scatter(L_np[noise_mask, 0], L_np[noise_mask, 1],
               c=COLOR_LOS, s=50, label='LOS')
    if T_np is not None:
        ax.scatter(T_np[:, 0], T_np[:, 1],
                   c=COLOR_GT, s=50, marker='o', label='NLOS GT')
    ax.scatter(T_pred_np[:, 0], T_pred_np[:, 1],
               c=COLOR_PRED, s=50, marker='+', label='NLOS Est.')

    for line_id, inlier_idx in line_inliers_map.items():
        pts      = L_plot[inlier_idx]
        a, b, c  = line_params_map[line_id]
        ax.scatter(pts[:, 0], pts[:, 1], color=COLOR_LOS, s=50)
        xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
        ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
        if np.abs(a) > np.abs(b):
            ax.plot([(-b * ymin - c) / (a + 1e-8),
                     (-b * ymax - c) / (a + 1e-8)],
                    [ymin, ymax],
                    color=COLOR_WALL, linestyle='--', linewidth=3)
        else:
            ax.plot([xmin, xmax],
                    [(-a * xmin - c) / (b + 1e-8),
                     (-a * xmax - c) / (b + 1e-8)],
                    color=COLOR_WALL, linestyle='--', linewidth=3)
    if line_params_map:
        ax.plot([], [], color=COLOR_WALL, linestyle='--', label='Est. Wall')

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlabel('X (m)')
    ax.set_yticks([])
    ax.legend(fontsize=7)
    plt.savefig(fig_out, format='svg', bbox_inches='tight', transparent=True)
    plt.close(fig)
