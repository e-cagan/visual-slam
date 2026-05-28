"""
Module for evaluating visual odometry performance metrics.
"""

import numpy as np


def trajectory_length(traj):
    """
    Calculates the total path length of a trajectory.
    traj: (N, 3) numpy array
    """
    return np.linalg.norm(np.diff(traj, axis=0), axis=1).sum()


def align_scale(est, gt):
    """
    Aligns the scale of the estimated trajectory to the ground-truth.
    Returns the scaled trajectory and the calculated scale factor.
    """
    len_gt = trajectory_length(gt)
    len_est = trajectory_length(est)
    
    scale = len_gt / (len_est + 1e-8)  # Added a little epsilon to prevent zero division error
    est_aligned = est * scale
    
    return est_aligned, scale


def compute_ate(est, gt):
    """
    Computes the Absolute Trajectory Error (RMSE) after scale alignment.
    Returns: ATE (float), est_aligned (N,3)
    """
    # Align the scale
    est_aligned, scale = align_scale(est, gt)
    
    # Find the difference between 2 matrixes
    diffs = est_aligned - gt
    
    # Calculate RMSE (Root Mean Square Error)
    ate = np.sqrt(np.mean(np.sum(diffs**2, axis=1)))
    
    return ate, scale