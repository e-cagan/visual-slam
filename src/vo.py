"""
Module for visual odometry.
"""

import numpy as np
import cv2
from .features import extract_features, match_features, get_matched_points
from .geometry import triangulate, solve_pnp


class VisualOdometry:
    """
    Stereo Visual Odometry class using Frame-to-Frame Tracking (Interval=1).
    """
    # We are defaulting keyframe_interval to 1 to enforce Frame-to-Frame tracking.
    # This prevents the local map from decaying rapidly, significantly improving scale.
    def __init__(self, dataset, keyframe_interval=1):
        self.ds = dataset
        self.K = dataset.K
        self.P_left = dataset.P_left
        self.P_right = dataset.P_right
        self.keyframe_interval = keyframe_interval
        
        # Global trajectory history -> (N, 3)
        self.trajectory = []  

        # Map state — built map based on the last keyframe
        self.map_points = None                # (M, 3) — 3D points, LAST KEYFRAME coord. system
        self.map_descriptors = None           # (M, 32) — corresponding left camera descriptors
        self.T_keyframe_to_world = np.eye(4)  # Last keyframe's pose with respect to the world

    def process(self):
        """
        Runs the full VO pipeline over the dataset.
        """
        self.trajectory = []
        
        # Initialize the first frame as the first keyframe (Origin)
        self._build_map(frame_idx=0)
        T_global = np.eye(4)
        self.trajectory.append(T_global[:3, 3].copy())
        
        total_frames = len(self.ds)
        print(f"\n--- Starting Stereo VO (Total Frames: {total_frames}) ---")
        
        for i in range(1, total_frames):
            # Track: Where is the camera relative to the last keyframe?
            T_curr_to_kf = self._track(i)
            
            # Establish global pose: keyframe's global pose x camera's relative pose
            T_global = self.T_keyframe_to_world @ T_curr_to_kf
            self.trajectory.append(T_global[:3, 3].copy())
            
            # Keyframe update criterion (Interval=1 means EVERY frame is a keyframe)
            if i % self.keyframe_interval == 0:
                self.T_keyframe_to_world = T_global.copy()
                self._build_map(frame_idx=i)
                
            # Progress print
            if i % 20 == 0 or i == total_frames - 1:
                progress = (i / (total_frames - 1)) * 100
                print(f"Tracking Progress: {i}/{total_frames - 1} [{progress:.1f}%]")
        
        print("Visual Odometry processing complete!\n")
        return np.array(self.trajectory)
    
    def _build_map(self, frame_idx):
        """
        Triangulates stereo pairs and stores 3D points & descriptors.
        Side effect: Updates self.map_points and self.map_descriptors.
        """
        # Extract features + match + get matched points
        img_L, img_R = self.ds.get_stereo_frame(frame_idx)
        kp_L, desc_L = extract_features(img_L)
        kp_R, desc_R = extract_features(img_R)
        
        matches = match_features(desc_L, desc_R)
        pts_L, pts_R = get_matched_points(kp_L, kp_R, matches)
        
        # Triangulate (Z > 0 and Z < 100 filter applies inside)
        pts_3d_valid, valid_mask = triangulate(pts_L, pts_R, self.P_left, self.P_right, max_depth=100)
        
        # Descriptor sync: collect left descriptors using the matches sequence
        matched_desc_L = np.array([desc_L[m.queryIdx] for m in matches])  # (N, 32)
        matched_desc_L = matched_desc_L[valid_mask]                       # (M, 32)
        
        self.map_points = pts_3d_valid
        self.map_descriptors = matched_desc_L
    
    def _track(self, frame_idx):
        """
        Solves camera pose against the local map using PnP.
        Returns: T_curr_to_keyframe (4×4) — camera pose relative to the last keyframe.
        """
        # Read the current left image and extract features
        img_L = cv2.imread(self.ds.frame_paths[frame_idx], cv2.IMREAD_GRAYSCALE)
        kp_curr, desc_curr = extract_features(img_L)
        
        # Map ↔ current frame matching
        matches = match_features(self.map_descriptors, desc_curr)
        
        # queryIdx → map index (3D), trainIdx → curr frame keypoint (2D)
        # MUST BE FLOAT32 FOR OPENCV PnP
        obj_pts = np.array([self.map_points[m.queryIdx] for m in matches], dtype=np.float32)    
        img_pts = np.array([kp_curr[m.trainIdx].pt for m in matches], dtype=np.float32)         
        
        R, t, inliers = solve_pnp(obj_pts, img_pts, self.K)
        
        # Tracking check: if PnP fails or has too few inliers, return Identity (No movement)
        if R is None or inliers is None or len(inliers) < 10:
            print(f"Warning: Tracking lost at frame {frame_idx}! (Inliers: {len(inliers) if inliers is not None else 0})")
            return np.eye(4)
        
        # PnP returns World->Camera (W→C) pose. We need Camera->World (C→W).
        # Build 4x4 matrix and invert it.
        T_pnp = np.eye(4)
        T_pnp[:3, :3] = R
        T_pnp[:3, 3] = t.ravel()
        T_curr_to_keyframe = np.linalg.inv(T_pnp)
        
        return T_curr_to_keyframe