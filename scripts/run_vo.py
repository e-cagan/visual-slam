"""
Module for visual odometry running script.
"""

import sys
import os

# Main path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
from src.dataset import KittiDataset
from src.vo import VisualOdometry
from src.visualization import plot_trajectory, plot_trajectories, draw_matched_features
from src.features import extract_features, match_features, get_matched_points
from src.geometry import estimate_essential_matrix, recover_pose
from src.metrics import compute_ate_aligned, compute_ate_raw


if __name__ == '__main__':
    # Load dataset and Ground-Truth (GT)
    ds = KittiDataset(sequence_id="04", base_path="data/dataset")
    gt_trajectory = ds.get_trajectory()
    
    # Visual Odometry
    # Full pipeline
    print("\n--- Full M2 pipeline ---")
    vo = VisualOdometry(ds, keyframe_interval=1)
    est_traj = vo.process()
    gt_traj = ds.get_trajectory()

    # İki ATE: raw (scale alignment YOK) + scale-aligned (M1'le karşılaştırma için)
    ate_raw = compute_ate_raw(est_traj, gt_traj)
    ate_aligned, scale = compute_ate_aligned(est_traj, gt_traj)
    print(f"Scale factor: {scale:.4f}  (M2 beklenen: ~1.0)")
    print(f"Raw ATE (no scale align): {ate_raw:.4f} m")
    print(f"Aligned ATE: {ate_aligned:.4f} m  (M1: 8.74m)")

    plot_trajectories(est_traj, gt_traj, title="M2: Stereo + PnP")