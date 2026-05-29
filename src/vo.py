"""
Module for visual odometry.
"""

import numpy as np
import cv2
from .features import extract_features, match_features, get_matched_points
from .geometry import estimate_essential_matrix, recover_pose


class VisualOdometry:
    """
    Visual Odometry class.
    """
    
    def __init__(self, dataset):
        self.ds = dataset
        self.K = dataset.K
        self.trajectory = []  # global positions -> (N, 3)

    def process(self):
        # Create 4x4 unit matrix
        T_global = np.eye(4)
        
        # Add initial position to trajectory
        self.trajectory.append(T_global[:3, 3].copy())
        
        # Read first frame's features for optimization
        print("Reading Frame 0...")
        img_prev = cv2.imread(self.ds.frame_paths[0], cv2.IMREAD_GRAYSCALE)
        kp_prev, desc_prev = extract_features(img_prev)

        # Iterate trough consecutive frames
        for i in range(1, len(self.ds)):
            # Read current frame and extract the features of it
            img_curr = cv2.imread(self.ds.frame_paths[i], cv2.IMREAD_GRAYSCALE)
            kp_curr, desc_curr = extract_features(img_curr)

            # Match the features between previous and current
            matches = match_features(desc_prev, desc_curr)
            pts_prev, pts_curr = get_matched_points(kp_prev, kp_curr, matches)

            # Calculate essential matrix and RANSAC inlier mask
            E, mask = estimate_essential_matrix(pts_prev, pts_curr, self.K)

            # Calculate relative pose
            R_rel, t_rel = recover_pose(E, pts_prev, pts_curr, self.K, mask=mask.copy())

            # Build 4x4 transformation matrix
            T_rel = np.eye(4)
            T_rel[:3, :3] = R_rel           # top left 3x3 rotation
            T_rel[:3, 3] = t_rel.flatten()  # top right 3x1 translation (fixed the shape with flatten)
            
            # Take the inverse of transformation matrix to find camera to world position
            T_rel_inv = np.linalg.inv(T_rel)
            T_global = T_global @ T_rel_inv

            # Eğer yukarıdaki konvansiyon rotayı tamamen yanlış (ters/aşağı) çizerse, şunu deneyeceğiz:
            # T_global = T_global @ T_rel

            # Append the new global (x, y, z) coordinates to trajectory
            self.trajectory.append(T_global[:3, 3].copy())

            # Update previous frames
            kp_prev, desc_prev = kp_curr, desc_curr

        # Convert trajectory to numpy array
        print("Visual Odometry processing complete!")
        
        return np.array(self.trajectory)