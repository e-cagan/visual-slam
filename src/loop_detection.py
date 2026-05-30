"""
Module for loop closure detection.

Two-stage: 
1. Appearance-based candidate selection (descriptor similarity).
2. Geometric verification (relative pose between candidate and current keyframe).
"""

import numpy as np
import cv2
from .features import extract_features, match_features
from .geometry import solve_pnp


class LoopDetector:
    """
    Detects when the camera revisits a previously seen place.
    """
    
    def __init__(self, map_obj, min_keyframe_gap=50, similarity_threshold=0.15, 
                 min_inliers=30):
        """
        Args:
            map_obj: reference to the persistent Map (from mapping.py)
            min_keyframe_gap: ignore candidates with id closer than this 
                              (temporal exclusion — recent keyframes always look similar)
            similarity_threshold: minimum normalized similarity to be a candidate
            min_inliers: minimum PnP inliers for geometric verification to pass
        """
        self.map = map_obj
        self.min_keyframe_gap = min_keyframe_gap
        self.similarity_threshold = similarity_threshold
        self.min_inliers = min_inliers
    
    def detect(self, current_kf_id):
        """
        For the given current keyframe, check if it closes a loop.
        
        Returns:
            (matched_kf_id, T_curr_to_matched) if loop found, else None.
            T_curr_to_matched: 4x4 transform — current keyframe in matched keyframe's frame.
        """
        # 1. Get all eligible past keyframes (id < current - min_gap)
        # 2. For each candidate, compute appearance similarity
        # 3. Take top-N candidates above threshold
        # 4. For each, do geometric verification (match + PnP)
        # 5. Return first one that passes, else None
        ...
    
    def _compute_similarity(self, kf_id_a, kf_id_b):
        """
        Appearance similarity between two keyframes.
        
        Simple version: ratio of mutual descriptor matches between keyframe a and b.
        Production version: BoW histogram dot product (requires DBoW2 vocabulary).
        
        Returns: float in [0, 1]
        """
        result = kf_id_a / kf_id_b
        return np.clip(result, a_min=0.0, a_max=1.0)
    
    def _geometric_verification(self, kf_curr, kf_candidate):
        """
        Try to compute relative pose between two keyframes using their 3D-2D 
        correspondences. If enough inliers, loop is real.
        
        Steps:
        1. Match descriptors between kf_curr's observed points and kf_candidate's image
        2. Use kf_candidate's 3D map points as object points, kf_curr's 2D as image points
        3. PnP RANSAC → if inliers > min_inliers, return relative pose
        
        Returns: T_curr_to_candidate (4x4) if verified, else None
        """
        ...