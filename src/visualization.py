"""
Module for visualizations.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import cv2


def plot_trajectory(traj, title="Ground-truth trajectory"):
    """Plots traj: (N, 3). plots X-Z plane: x axis = traj[:,0], y axis = traj[:,2]; ty (traj[:,1]) = height"""
    # Create results dir to avoid error (skip if already present)
    os.makedirs("results", exist_ok=True)
    
    # Plot info
    plt.title(title)
    plt.plot(traj[:, 0], traj[:, 2])
    plt.axis('equal')
    plt.xlabel("X")
    plt.ylabel("Z")
    
    # Plot saving
    plt.savefig("results/trajectory.png")
    print("Trajectory plot has been saved successfully to 'results/trajectory.png'.")
    plt.show()


def plot_trajectories(est_traj, gt_traj, title="Estimated vs Ground-truth", align_scale=False):
    """
    Plots two trajectories on the same graph for comparison.
    If align_scale is True, it roughly scales the estimated trajectory to match GT size for visual comparison.
    """
    # Create results dir to avoid error (skip if already present)
    os.makedirs("results", exist_ok=True)
    
    # Scale alignment to compare visualizations
    if align_scale:
        s_gt = np.linalg.norm(np.diff(gt_traj, axis=0), axis=1).sum()
        s_est = np.linalg.norm(np.diff(est_traj, axis=0), axis=1).sum()
        scale_factor = s_gt / (s_est + 1e-6) # Zero division protection
        est_plot = est_traj * scale_factor
    else:
        est_plot = est_traj

    plt.figure(figsize=(10, 8))
    plt.title(title)
    
    # Ground Truth plot
    plt.plot(gt_traj[:, 0], gt_traj[:, 2], label='Ground Truth', color='black', linestyle='--')
    
    # Estimated plot
    plt.plot(est_plot[:, 0], est_plot[:, 2], label='Estimated (VO)', color='blue')
    
    plt.axis('equal')
    plt.xlabel("X (meters)")
    plt.ylabel("Z (meters)")
    plt.legend()
    plt.grid(True)
    
    plt.savefig("results/comparison_trajectory.png")
    print("Comparison plot has been saved successfully to 'results/comparison_trajectory.png'.")
    plt.show()


def draw_matched_features(img1, img2, kp1, kp2, matches, num_matches=50):
    """
    Draws matched features on images
    """
    # Draw matched feature limited by "num_matches" and save it
    drawn_img = cv2.drawMatches(
        img1, kp1, 
        img2, kp2, 
        matches[:num_matches], 
        None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite("results/matches.png", drawn_img)
    print("Matched feature plot saved to 'results/matches.png'.")
    cv2.imshow('Feature matches', drawn_img)
    
    # Doesn't quit until a key pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return drawn_img