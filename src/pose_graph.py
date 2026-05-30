"""
Module for pose graph optimization after loop closure detection.

Uses GTSAM for pose-only optimization (does not optimize 3D points — that 
would be full BA, much heavier and out of scope for M4).
"""

import gtsam
import numpy as np
from gtsam.symbol_shorthand import X


def build_pose_graph(map_obj, loop_closures):
    """
    Construct pose graph from current map state.
    
    Args:
        map_obj: persistent Map (from mapping.py)
        loop_closures: list of (kf_curr_id, kf_matched_id, T_pnp_4x4) tuples
    
    Returns:
        (graph, initial_estimate): gtsam objects ready for optimization
    """
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()
    
    kf_ids = sorted(map_obj.keyframes.keys())
    
    # Prior on first keyframe (anchor the world origin)
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6]*6))
    first_pose = gtsam.Pose3(map_obj.keyframes[kf_ids[0]].pose)
    graph.add(gtsam.PriorFactorPose3(X(kf_ids[0]), first_pose, prior_noise))
    
    # Initial estimates from current keyframe poses
    for kf_id in kf_ids:
        pose = gtsam.Pose3(map_obj.keyframes[kf_id].pose)
        initial_estimate.insert(X(kf_id), pose)
    
    # Sequential odometry edges (consecutive keyframe pairs)
    odom_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.05, 0.05, 0.05, 0.1, 0.1, 0.1])
    )
    for i in range(len(kf_ids) - 1):
        id_a, id_b = kf_ids[i], kf_ids[i+1]
        pose_a = gtsam.Pose3(map_obj.keyframes[id_a].pose)
        pose_b = gtsam.Pose3(map_obj.keyframes[id_b].pose)
        T_ab = pose_a.between(pose_b)
        graph.add(gtsam.BetweenFactorPose3(X(id_a), X(id_b), T_ab, odom_noise))
    
    # Loop closure edges (less confident than sequential)
    loop_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3])
    )
    for (kf_curr_id, kf_matched_id, T_pnp) in loop_closures:
        T_relative = gtsam.Pose3(T_pnp)
        graph.add(gtsam.BetweenFactorPose3(
            X(kf_matched_id), X(kf_curr_id), T_relative, loop_noise
        ))
    
    return graph, initial_estimate


def optimize_pose_graph(graph, initial_estimate, max_iterations=50):
    """
    Run Levenberg-Marquardt optimization on the pose graph.
    
    Returns:
        dict {kf_id: 4x4 numpy pose} with optimized poses
    """
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(max_iterations)
    params.setVerbosity("SILENT")
    
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    
    # Extract optimized poses, keyed by keyframe id
    optimized_poses = {}
    for key in result.keys():
        sym = gtsam.Symbol(key)
        kf_id = sym.index()
        optimized_poses[kf_id] = result.atPose3(key).matrix()
    
    return optimized_poses


def apply_optimized_poses(map_obj, optimized_poses):
    """Update keyframe poses in place after optimization."""
    for kf_id, pose_4x4 in optimized_poses.items():
        if kf_id in map_obj.keyframes:
            map_obj.keyframes[kf_id].pose = pose_4x4