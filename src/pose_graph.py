"""
Module for pose graph optimization after loop closure.

Uses g2o-python or gtsam under the hood.
Optimizes all keyframe poses subject to:
- Sequential edges (relative poses between consecutive keyframes)
- Loop closure edges (relative poses between keyframes that observe the same place)

Does NOT optimize 3D points (that would be full BA, much heavier).
"""

import numpy as np
import gtsam


def build_pose_graph(map_obj, loop_closures):
    """
    Construct pose graph from current map state and detected loops.
    
    Args:
        map_obj: persistent Map
        loop_closures: list of (kf_id_a, kf_id_b, T_a_to_b) tuples
    
    Returns:
        graph object (gtsam.NonlinearFactorGraph or g2o equivalent)
        initial values
    """
    # 1. Create empty graph + initial estimates
    # 2. Add prior on first keyframe (fix world origin)
    # 3. For each consecutive keyframe pair: add BetweenFactor with relative pose
    # 4. For each loop closure: add BetweenFactor with loop pose
    # 5. Return graph + initial values
    ...


def optimize_pose_graph(graph, initial_values, max_iterations=20):
    """
    Run Levenberg-Marquardt on the pose graph.
    
    Returns:
        optimized_poses: dict {kf_id: 4x4 pose}
    """
    # Use gtsam.LevenbergMarquardtOptimizer or g2o equivalent
    # Run optimization
    # Extract final poses
    ...


def apply_optimized_poses(map_obj, optimized_poses):
    """
    Update keyframe poses and (optionally) re-transform 3D points based on 
    their host keyframe's old vs new pose.
    
    Two strategies for point updates:
    A) Don't update points — accept that they're now slightly inconsistent.
    B) For each point: transform from old host pose to new host pose.
    
    For M4, A is fine. Full BA (M5+) handles point updates properly.
    """
    ...