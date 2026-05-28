"""
Module for visual odometry running script.
"""

import sys
import os

# Main path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
from src.dataset import KittiDataset
from src.vo import VisualOdometry
from src.visualization import plot_trajectory, plot_trajectories, draw_matched_features
from src.features import extract_features, match_features, get_matched_points
from src.geometry import estimate_essential_matrix, recover_pose
from src.metrics import compute_ate


if __name__ == '__main__':
    # Load dataset and Ground-Truth (GT)
    ds = KittiDataset(sequence_id="04", base_path="data/dataset")
    gt_trajectory = ds.get_trajectory()
    
    # Visual Odometry
    vo = VisualOdometry(ds)
    estimated_trajectory = vo.process()
    
    # Metric calculation
    ate, scale_factor = compute_ate(estimated_trajectory, gt_trajectory)
    
    print("\n--- Final Metrics ---")
    print(f"Calculated Scale Factor: {scale_factor:.4f}")
    print(f"Absolute Trajectory Error (ATE): {ate:.4f} meters")
    
    # Plotting
    plot_trajectories(estimated_trajectory, gt_trajectory, align_scale=True)