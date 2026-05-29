"""
Module for visual odometry running script.
"""

import sys
import os

# Main path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.dataset import KittiDataset
from src.vo import VisualOdometry
from src.visualization import plot_trajectories
from src.metrics import compute_ate_aligned, compute_ate_raw

if __name__ == '__main__':
    # 1. Dataset and Ground-Truth (GT) loading
    ds = KittiDataset(sequence_id="04", base_path="data/dataset")
    gt_traj = ds.get_trajectory()
    
    # Run tracking on every frame for maximum stability
    kf_interval = 1 

    print("\n===========================================")
    print("          M3: VISUAL ODOMETRY RUN          ")
    print("===========================================\n")

    # 2. Initialize and run Visual Odometry
    vo = VisualOdometry(ds, keyframe_interval=kf_interval)
    est_traj = vo.process()
    
    # 3. Calculate metrics
    ate_raw = compute_ate_raw(est_traj, gt_traj)
    _, scale = compute_ate_aligned(est_traj, gt_traj)
    
    # 4. Report results
    print("\n===========================================")
    print("             FINAL M3 METRICS              ")
    print("===========================================")
    print(f"Scale Factor: {scale:.4f}")
    print(f"Raw ATE     : {ate_raw:.4f} meters")
    
    # 5. Visualization
    plot_trajectories(est_traj, gt_traj, title="M3: Persistent Map VO vs GT")