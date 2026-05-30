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
    
    def __init__(self, map_obj, K, min_keyframe_gap=50, similarity_threshold=0.15, 
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
        self.K = K
        self.min_keyframe_gap = min_keyframe_gap
        self.similarity_threshold = similarity_threshold
        self.min_inliers = min_inliers
    
    def detect(self, current_kf_id):
        """
        Check if current keyframe closes a loop with any past keyframe.
        
        Returns:
            (matched_kf_id, T_curr_to_matched) if loop confirmed, else None.
        """
        # Eligible candidates: temporal exclusion
        # (current'tan en az min_keyframe_gap kadar geride olan keyframe'ler)
        all_kf_ids = sorted(self.map.keyframes.keys())
        eligible_ids = [
            kf_id for kf_id in all_kf_ids 
            if kf_id < current_kf_id - self.min_keyframe_gap
        ]
        
        if not eligible_ids:
            return None
        
        # Appearance similarity for each candidate
        candidates = []
        for kf_id in eligible_ids:
            sim = self._compute_similarity(current_kf_id, kf_id)
            if sim >= self.similarity_threshold:
                candidates.append((kf_id, sim))
        
        if not candidates:
            return None
        
        # Sort by similarity (descending), take top-N
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidates[:3]  # top 3
        
        # Geometric verification — first that passes wins
        for cand_id, sim in top_candidates:
            T_curr_to_cand = self._geometric_verification(current_kf_id, cand_id)
            
            if T_curr_to_cand is not None:
                print(f"Loop closure: KF {current_kf_id} ↔ KF {cand_id} (sim={sim:.3f})")
                return cand_id, T_curr_to_cand
        
        # Nothing passed verification
        return None
    
    def _compute_similarity(self, kf_id_a, kf_id_b):
        """
        Appearance similarity between two keyframes.
        
        Simple version: ratio of mutual descriptor matches between keyframe a and b.
        Production version: BoW histogram dot product (requires DBoW2 vocabulary).
        
        Returns: float in [0, 1]
        """
        desc_a = self._get_keyframe_descriptors(kf_id_a)  # All 3D point descriptors
        desc_b = self._get_keyframe_descriptors(kf_id_b)
        
        if len(desc_a) < 20 or len(desc_b) < 20:
            return 0.0
        
        # BFMatcher Hamming with cross check
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc_a, desc_b)
        
        # Distance threshold ile kötü match'leri ele
        good = [m for m in matches if m.distance < 50]
        
        # Similarity = good matches / min(set size)
        sim = len(good) / min(len(desc_a), len(desc_b))
        return sim
    
    def _geometric_verification(self, kf_curr_id, kf_candidate_id):
        """
        Try to compute relative pose between two keyframes using their 3D-2D 
        correspondences. If enough inliers, loop is real.
        
        Steps:
        1. Match descriptors between kf_curr's observed points and kf_candidate's image
        2. Use kf_candidate's 3D map points as object points, kf_curr's 2D as image points
        3. PnP RANSAC → if inliers > min_inliers, return relative pose
        
        Returns: T_curr_to_candidate (4x4) if verified, else None
        """
        # Current and candidate keyframes
        kf_curr = self.map.keyframes[kf_curr_id]
        kf_cand = self.map.keyframes[kf_candidate_id]
        
        # Candidate's 3D points (world coordinates) + their descriptors
        cand_pts_3d = []
        cand_descs = []
        for pt_id in kf_cand.observed_point_ids:
            pt = self.map.points[pt_id]
            cand_pts_3d.append(pt.position)
            cand_descs.append(pt.descriptor)
        
        if len(cand_pts_3d) < self.min_inliers:
            return None
        
        cand_pts_3d = np.array(cand_pts_3d, dtype=np.float32)
        cand_descs = np.array(cand_descs, dtype=np.uint8)
        
        # Current keyframe's features — already stored, no image re-read needed
        kp_curr = kf_curr.keypoints
        desc_curr = kf_curr.descriptors
        
        # Match candidate's 3D point descriptors ↔ current keyframe's 2D descriptors
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(cand_descs, desc_curr)
        
        if len(matches) < self.min_inliers:
            return None
        
        obj_pts = np.array([cand_pts_3d[m.queryIdx] for m in matches], dtype=np.float32)
        img_pts = np.array([kp_curr[m.trainIdx].pt for m in matches], dtype=np.float32)
        
        R, t, inliers = solve_pnp(obj_pts, img_pts, self.K)
        
        if R is None or inliers is None or len(inliers) < self.min_inliers:
            return None
        
        T_pnp = np.eye(4)
        T_pnp[:3, :3] = R
        T_pnp[:3, 3] = t.ravel()
        
        return T_pnp
    
    def _get_keyframe_descriptors(self, kf_id):
        """Return descriptors of a keyframe (from its image features)."""
        return self.map.keyframes[kf_id].descriptors