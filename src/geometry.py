"""
Module for image geometry.
"""

import cv2
import numpy as np


def estimate_essential_matrix(pts1, pts2, K):
    """
    Solve E with RANSAC. 
    Returns: E (3,3), mask (inlier signs)
    """
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    return E, mask


def recover_pose(E, pts1, pts2, K, mask):
    """
    Decompose E to R, t. Selects the right solution with cheirality test using ONLY RANSAC inliers.
    Returns: R (3,3), t (3,1) — t Unit NORM
    """
    _, R, t, mask_out = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    return R, t