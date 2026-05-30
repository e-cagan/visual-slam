"""
Module for visual odometry with persistent map and local BA.
"""

import numpy as np
import cv2
from .features import extract_features, match_features, get_matched_points
from .pose_graph import build_pose_graph, optimize_pose_graph, apply_optimized_poses
from .geometry import triangulate, solve_pnp
from .loop_detection import LoopDetector
from .mapping import Map


class VisualOdometry:
    """
    Stereo Visual Odometry class with a Persistent Map and Bundle Adjustment.
    """
    
    def __init__(self, dataset, keyframe_interval=5, enable_loop_closure=True):
        self.ds = dataset
        self.K = dataset.K
        self.P_left = dataset.P_left
        self.P_right = dataset.P_right
        
        self.keyframe_interval = keyframe_interval
        self.enable_loop_closure = enable_loop_closure
        
        # M3: The persistent map replaces the temporary variables
        self.map = Map()                     
        self.trajectory = []
        self.loop_closures = []
        self.last_keyframe_id = None
        self.loop_detector = None
    
    def process(self):
        """
        Runs the full VO pipeline over the dataset.
        """
        self.trajectory = []
        
        print("\n--- Starting Stereo VO (M3: Persistent Map + BA) ---")
        
        # Initialize the first frame as the first keyframe
        self._initialize(frame_idx=0)
        
        for i in range(1, len(self.ds)):
            # Tracking: localize the new frame against the existing map using PnP + BA
            track_result = self._track(i)
            
            tracking_lost = False # Takip durumunu tutacak bayrak
            
            if track_result is None:
                print(f"Warning: Tracking lost at frame {i}. Using last known pose.")
                T_world_to_cam = self.map.keyframes[self.last_keyframe_id].pose.copy()
                tracked_point_ids = []
                matched_kps = []
                tracking_lost = True # Takip koptu!
            else:
                T_world_to_cam, tracked_point_ids, matched_kps = track_result
            
            # Extract camera position
            T_cam_to_world = np.linalg.inv(T_world_to_cam)
            cam_position = T_cam_to_world[:3, 3]
            self.trajectory.append(cam_position.copy())
            
            # Don't add keyframe if tracking is lost
            if not tracking_lost and self._should_add_keyframe(i, T_world_to_cam):
                self._add_keyframe(i, T_world_to_cam, tracked_point_ids, matched_kps)

                # Loop closure detection
                if self.enable_loop_closure and self.last_keyframe_id > 50:
                    result = self.loop_detector.detect(self.last_keyframe_id)
                    if result is not None:
                        matched_kf_id, T_loop = result
                        self.loop_closures.append(
                            (self.last_keyframe_id, matched_kf_id, T_loop)
                        )
                        print(f"Loop detected: KF {self.last_keyframe_id} ↔ KF {matched_kf_id}")
                        
                        # Pose graph optimization
                        self._optimize_pose_graph()
                        
                        # Rebuild the trajectory since the keyframe poses changed
                        self._rebuild_trajectory()
            
            # Progress print
            if i % 20 == 0 or i == len(self.ds) - 1:
                progress = (i / (len(self.ds) - 1)) * 100
                print(f"Tracking Progress: {i}/{len(self.ds) - 1} [{progress:.1f}%]")
        
        print("Visual Odometry processing complete!\n")
        return np.array(self.trajectory)
    
    def _optimize_pose_graph(self):
        """Build + optimize + apply pose graph."""
        graph, initial = build_pose_graph(self.map, self.loop_closures)
        optimized = optimize_pose_graph(graph, initial)
        apply_optimized_poses(self.map, optimized)
    
    def _rebuild_trajectory(self):
        """Pose graph optimization sonrası trajectory'yi keyframe pozlarından yeniden kur."""
        # Şu an trajectory her frame için kaydedildi, ama keyframe'ler arası kareler için 
        # interpolasyon ya da basitçe sadece keyframe pozlarını kullan
        ...
    
    def _initialize(self, frame_idx):
        """
        Establishes the first keyframe and builds the initial map.
        """
        # Read stereo frame, extract features, and match
        img_L, img_R = self.ds.get_stereo_frame(frame_idx)
        kp_L, desc_L = extract_features(img_L)
        kp_R, desc_R = extract_features(img_R)
        
        matches = match_features(desc_L, desc_R)
        pts_L, pts_R = get_matched_points(kp_L, kp_R, matches)
        
        # Triangulate to get 3D points
        pts_3d_valid, valid_mask = triangulate(pts_L, pts_R, self.P_left, self.P_right, max_depth=100)
        
        # Filter descriptors and keypoints using the valid mask
        valid_desc_L = np.array([desc_L[m.queryIdx] for m in matches])[valid_mask]
        valid_kp_L = pts_L[valid_mask]
        
        # Add the initial keyframe to the map (Pose is Identity: W->C)
        initial_pose = np.eye(4)
        kf_id = self.map.add_keyframe(initial_pose)
        self.last_keyframe_id = kf_id
        
        # Add all valid 3D points to the map and link them to the keyframe
        for i in range(len(pts_3d_valid)):
            self.map.add_point(
                position_3d=pts_3d_valid[i], 
                descriptor=valid_desc_L[i], 
                kf_id=kf_id, 
                pixel_2d=valid_kp_L[i]
            )
            
        # Add origin to trajectory
        self.trajectory.append(np.zeros(3))

        # Enable loop detector based on argument
        if self.enable_loop_closure:
            self.loop_detector = LoopDetector(self.map)

        print(f"Initialized map with {len(pts_3d_valid)} points at Frame {frame_idx}.")
    
    def _track(self, frame_idx):
        """
        Localizes the new frame against the existing map using PnP and Motion-Only BA.
        Returns: 
            T_world_to_cam (4x4), list of matched map point IDs, and their 2D pixel coordinates.
            Returns None if tracking fails.
        """
        # Extract features from the current left image
        img_L = cv2.imread(self.ds.frame_paths[frame_idx], cv2.IMREAD_GRAYSCALE)
        kp_curr, desc_curr = extract_features(img_L)
        
        # Get all active descriptors from the persistent map
        map_descs, map_pt_ids = self.map.get_active_descriptors(10)
        
        if len(map_descs) == 0:
            return None
            
        # Match map descriptors with current frame descriptors
        matches = match_features(map_descs, desc_curr)
        
        if len(matches) < 10:
            return None
            
        # Prepare data for PnP
        obj_pts = []
        img_pts = []
        tracked_pt_ids = []
        
        for m in matches:
            map_idx = m.queryIdx
            curr_idx = m.trainIdx
            
            pt_id = map_pt_ids[map_idx]
            obj_pts.append(self.map.points[pt_id].position)
            img_pts.append(kp_curr[curr_idx].pt)
            tracked_pt_ids.append(pt_id)
            
        obj_pts = np.array(obj_pts, dtype=np.float32)
        img_pts = np.array(img_pts, dtype=np.float32)
        
        # Solve PnP to get an initial pose guess
        R, t, inliers = solve_pnp(obj_pts, img_pts, self.K)
        
        if R is None or inliers is None or len(inliers) < 10:
            return None
            
        initial_pose = np.eye(4)
        initial_pose[:3, :3] = R
        initial_pose[:3, 3] = t.ravel()
        
        # Extract inliers to ensure stability for downstream steps
        inlier_idx = inliers.ravel()
        obj_pts_inliers = obj_pts[inlier_idx]
        img_pts_inliers = img_pts[inlier_idx]
        tracked_pt_ids_inliers = [tracked_pt_ids[i] for i in inlier_idx]
        
        T_pnp = initial_pose
            
        # Because our map is now in GLOBAL WORLD coordinates, T_pnp is exactly T_world_to_cam.
        # We return it directly without inverting it here.
        return T_pnp, tracked_pt_ids_inliers, img_pts_inliers
    
    def _should_add_keyframe(self, frame_idx, current_pose):
        """
        Determines if a new keyframe should be spawned.
        Simple heuristic: Every N frames.
        """
        return frame_idx % self.keyframe_interval == 0
    
    def _add_keyframe(self, frame_idx, T_world_to_cam, tracked_point_ids, matched_kps):
        """
        Spawns a new keyframe:
        - Registers the keyframe in the map.
        - Adds observations for existing tracked points.
        - Triangulates new stereo features to expand the map.
        """
        # Add the new keyframe to the map
        kf_id = self.map.add_keyframe(T_world_to_cam)
        self.last_keyframe_id = kf_id
        
        # Record observations for the points we successfully tracked
        for pt_id, pixel in zip(tracked_point_ids, matched_kps):
            self.map.add_observation(pt_id, kf_id, pixel)
            
        # Read stereo images to find NEW points (Map Expansion)
        img_L, img_R = self.ds.get_stereo_frame(frame_idx)
        kp_L, desc_L = extract_features(img_L)
        kp_R, desc_R = extract_features(img_R)
        
        matches = match_features(desc_L, desc_R)
        pts_L, pts_R = get_matched_points(kp_L, kp_R, matches)
        
        pts_3d_valid, valid_mask = triangulate(pts_L, pts_R, self.P_left, self.P_right, max_depth=100)
        
        valid_desc_L = np.array([desc_L[m.queryIdx] for m in matches])[valid_mask]
        valid_kp_L = pts_L[valid_mask]
        
        # Transform the newly triangulated points (which are in the local Camera frame)
        # into the Global World frame using the inverted pose.
        T_cam_to_world = np.linalg.inv(T_world_to_cam)
        
        new_points_added = 0
        for i in range(len(pts_3d_valid)):
            # Convert local 3D point to homogeneous, transform to world, and convert back
            pt_local_hom = np.append(pts_3d_valid[i], 1.0)
            pt_world_hom = T_cam_to_world @ pt_local_hom
            pt_world = pt_world_hom[:3] / pt_world_hom[3]
            
            # Add the new point to the map
            self.map.add_point(
                position_3d=pt_world, 
                descriptor=valid_desc_L[i], 
                kf_id=kf_id, 
                pixel_2d=valid_kp_L[i]
            )
            new_points_added += 1
            
        # print(f"Added Keyframe {kf_id} | Tracked {len(tracked_point_ids)} pts | Spawned {new_points_added} new pts.")