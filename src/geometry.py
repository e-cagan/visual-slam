"""
Module for image geometry.
"""

import cv2
import numpy as np
from scipy.optimize import least_squares


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


def reproject_point(point_3d, pose_4x4, K):
    """
    Projects a 3D point to a 2D pixel coordinate using the camera pose.
    
    Args:
        point_3d: (3,) numpy array representing the 3D point in world coordinates.
        pose_4x4: (4, 4) numpy array representing the world-to-camera transformation.
        K: (3, 3) intrinsic camera matrix.
        
    Returns:
        (2,) numpy array representing the estimated [u, v] pixel coordinate.
    """
    # Convert 3D point to homogeneous coordinates: [X, Y, Z, 1]
    point_3d_hom = np.append(point_3d, 1.0)
    
    # Transform the point from World to Camera coordinate system
    point_cam_hom = pose_4x4 @ point_3d_hom
    
    # Guard: If the point is behind the camera (Z <= 0), return an invalid pixel
    if point_cam_hom[2] <= 0:
        return np.array([-1.0, -1.0]) 
    
    # Normalize by Z (Zc) to project onto the normalized image plane
    point_cam_norm = point_cam_hom[:3] / point_cam_hom[2]
    
    # Apply the camera intrinsic matrix to get final pixel coordinates
    pixel_hom = K @ point_cam_norm
    
    return pixel_hom[:2]


def reprojection_error(params, points_3d, observed_pixels, K):
    """
    Calculates the reprojection error for all observations. 
    This function is iteratively called by scipy.optimize.least_squares.
    
    Args:
        params: (6,) flat array [rvec (3,), tvec (3,)].
        points_3d: (N, 3) fixed 3D points from the map.
        observed_pixels: (N, 2) actual 2D pixel observations in the current frame.
        K: (3, 3) intrinsic camera matrix.
        
    Returns:
        (2N,) flat array containing the [du, dv] errors for each observation.
    """
    rvec = params[:3]
    tvec = params[3:]
    
    # Convert Rodrigues rotation vector (3,) back to a 3x3 Rotation Matrix
    R, _ = cv2.Rodrigues(rvec)
    
    # Construct the 4x4 World-to-Camera pose matrix
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = tvec
    
    residuals = []
    
    # Calculate the error for each 3D point and its corresponding observation
    for pt_3d, obs_pixel in zip(points_3d, observed_pixels):
        
        # Reproject the 3D point onto the image plane using the estimated pose
        rep_pixel = reproject_point(pt_3d, pose, K)
        
        # Heavily penalize points that project behind the camera to guide the optimizer away
        if rep_pixel[0] < 0:
            error = np.array([1e6, 1e6]) 
        else:
            # The residual is the difference between the observed and reprojected pixels
            error = obs_pixel - rep_pixel
            
        residuals.append(error)
        
    # SciPy expects a flattened 1D array of residuals (length 2N)
    return np.concatenate(residuals)


def motion_only_ba(initial_pose, points_3d, observed_pixels, K):
    """
    Motion-only Bundle Adjustment. Optimizes ONLY the camera pose while keeping 
    the 3D map points fixed.
    
    Args:
        initial_pose: (4, 4) initial pose estimate (e.g., from PnP).
        points_3d: (N, 3) fixed 3D map points.
        observed_pixels: (N, 2) corresponding 2D pixel locations in the current frame.
        K: (3, 3) intrinsic camera matrix.
        
    Returns:
        refined_pose: (4, 4) optimized world-to-camera pose.
    """
    # Extract Rotation (R) and Translation (t) from the initial 4x4 guess
    R_init = initial_pose[:3, :3]
    t_init = initial_pose[:3, 3]
    
    # Convert the Rotation matrix to a Rodrigues vector
    rvec_init, _ = cv2.Rodrigues(R_init)
    rvec_init = rvec_init.ravel()
    
    # Flatten parameters into a single (6,) array: [rvec, tvec]
    initial_params = np.hstack((rvec_init, t_init))
    
    # Run the optimization using the Levenberg-Marquardt (lm) algorithm
    result = least_squares(
        reprojection_error, 
        initial_params, 
        method='lm', 
        args=(points_3d, observed_pixels, K),
        verbose=0  # Set to 2 if you want to see optimization logs during isolated tests
    )
    
    # Reconstruct the refined 4x4 pose matrix from the optimized parameters
    refined_params = result.x
    rvec_refined = refined_params[:3]
    tvec_refined = refined_params[3:]
    
    R_refined, _ = cv2.Rodrigues(rvec_refined)
    
    refined_pose = np.eye(4)
    refined_pose[:3, :3] = R_refined
    refined_pose[:3, 3] = tvec_refined
    
    return refined_pose