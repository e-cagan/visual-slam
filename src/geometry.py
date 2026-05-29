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


def triangulate(pts_left, pts_right, P_left, P_right, max_depth=100):
    """Extracts 3D points from stereo matches.
    pts_left, pts_right: (N, 2)
    
    P_left, P_right: (3, 4) projection matrices
    
    Returns: pts_3d (N, 3) — metric, in left camera coordinates"""
    # OpenCV expects (2, N) shape. So we transform the matricies
    pts_left_T = pts_left.T
    pts_right_T = pts_right.T
    
    # We triangulate: Result (4, N) dim homogeneous coordinates
    pts_4d_hom = cv2.triangulatePoints(P_left, P_right, pts_left_T, pts_right_T)
    
    # Convert homogeneous coordinates to 3D cartesian coordinates (x/w, y/w, z/w)
    # We take the transpose (Convert the shape to (N, 4)) to make the division easier
    pts_4d_hom = pts_4d_hom.T
    pts_3d = pts_4d_hom[:, :3] / pts_4d_hom[:, 3:]

    # Depth filter (Z > 0 and Z < max_depth)
    depths = pts_3d[:, 2]
    valid_mask = (depths > 0) & (depths < max_depth)

    # Filter out the valid depthed points
    pts_3d_valid = pts_3d[valid_mask]
    
    return pts_3d_valid, valid_mask


def solve_pnp(pts_3d, pts_2d, K, dist_coeffs=None):
    """Known 3D points + their 2D projections in new frame → camera pose.
    pts_3d: (N, 3) — 3D points in some reference frame (map)
    
    pts_2d: (N, 2) — pixel locations of these points in current frame
    
    K: (3, 3)
    
    Returns: R (3, 3), t (3, 1), inliers — METRIC scale (t is NOT unit norm)"""
    # Check for distortion coefficients
    if dist_coeffs is None:
        # KITTI dataset is distortion corrected. So the distortion is assumed 0
        dist_coeffs = np.zeros(5)

    # PnP + RANSAC
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts_3d, pts_2d, K, dist_coeffs, 
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success or inliers is None:
        return None, None, None
    
    # Convert rvec (Rotation vector) to rotation matrix (R)
    R, _ = cv2.Rodrigues(rvec)
    
    return R, tvec, inliers