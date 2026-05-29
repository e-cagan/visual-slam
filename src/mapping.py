"""
Module for persistent map management and bundle adjustment.

The most critical data structure layer of M3: 3D points and keyframes are
no longer discarded. They are accumulated and kept linked together. 
Bundle Adjustment (BA) operates on this interconnected structure.
"""

import numpy as np

class MapPoint:
    """
    A single 3D point in the map.
    
    Each point is observed in multiple keyframes — these observations
    are what BA optimizes against.
    """
    def __init__(self, point_id, position_3d, descriptor):
        self.id = point_id
        self.position = position_3d.astype(np.float64)  # (3,) — BA will update this
        self.descriptor = descriptor                    # (32,) uint8 — for matching
        self.observations = {}                          # {keyframe_id: 2d_pixel (2,)}
    
    def add_observation(self, keyframe_id, pixel_2d):
        """Records that this point was observed in a specific keyframe."""
        self.observations[keyframe_id] = pixel_2d
    
    def is_observed_by(self, keyframe_id):
        """Checks if this point was observed by the given keyframe."""
        return keyframe_id in self.observations


class Keyframe:
    """
    A keyframe: a camera pose + which map points it observed.
    
    Local BA optimizes these keyframes and the 3D points they observe together.
    """
    def __init__(self, keyframe_id, pose_4x4, image=None):
        self.id = keyframe_id
        self.pose = pose_4x4.astype(np.float64)  # 4x4, world-to-camera or camera-to-world
        self.observed_point_ids = set()          # IDs of MapPoints observed by this keyframe
        self.image = image                       # Optional, for debugging


class Map:
    """
    The persistent map: all keyframes + all 3D points + their observations.
    
    This replaces the temporary `self.map_points` used in M2.
    """
    def __init__(self):
        self.points = {}      # {point_id: MapPoint}
        self.keyframes = {}   # {keyframe_id: Keyframe}
        self.next_point_id = 0
        self.next_kf_id = 0
    
    def add_keyframe(self, pose_4x4, image=None):
        """Adds a new keyframe to the map and returns its ID."""
        kf_id = self.next_kf_id
        new_kf = Keyframe(kf_id, pose_4x4, image)
        self.keyframes[kf_id] = new_kf
        
        self.next_kf_id += 1
        return kf_id
    
    def add_point(self, position_3d, descriptor, kf_id, pixel_2d):
        """Adds a new 3D point and links it to the keyframe that observed it."""
        pt_id = self.next_point_id
        new_pt = MapPoint(pt_id, position_3d, descriptor)
        self.points[pt_id] = new_pt
        
        self.next_point_id += 1
        
        # Immediately create the bi-directional link between point and keyframe
        self.add_observation(pt_id, kf_id, pixel_2d)
        
        return pt_id
    
    def add_observation(self, point_id, kf_id, pixel_2d):
        """
        Records that an existing 3D point was observed by a keyframe.
        Maintains the double-entry bookkeeping required for BA.
        """
        if point_id in self.points and kf_id in self.keyframes:
            # 1. The point remembers the keyframe
            self.points[point_id].add_observation(kf_id, pixel_2d)
            # 2. The keyframe remembers the point
            self.keyframes[kf_id].observed_point_ids.add(point_id)
    
    def get_local_window(self, n_keyframes=5):
        """
        Returns the last N keyframes and all points observed by them.
        Local BA will run on this specific sub-graph.
        
        Returns: 
            local_keyframes: list of Keyframe objects
            local_points: list of MapPoint objects
        """
        # Get the IDs of the most recent N keyframes
        all_kf_ids = sorted(list(self.keyframes.keys()))
        local_kf_ids = all_kf_ids[-n_keyframes:] if len(all_kf_ids) > n_keyframes else all_kf_ids
        
        local_keyframes = [self.keyframes[k_id] for k_id in local_kf_ids]
        
        # Collect all unique points observed by these specific keyframes
        local_point_ids = set()
        for kf in local_keyframes:
            local_point_ids.update(kf.observed_point_ids)
            
        local_points = [self.points[p_id] for p_id in local_point_ids]
        
        return local_keyframes, local_points
    
    def get_active_descriptors(self, recent_keyframes=10):
        """Sadece son N keyframe'in gördüğü noktaları döndür.
        
        Returns: descriptors (M, 32), point_ids list"""
        all_kf_ids = sorted(self.keyframes.keys())
        recent_kf_ids = all_kf_ids[-recent_keyframes:]
        
        # Bu keyframe'lerin gördüğü tüm unique noktaları topla
        active_pt_ids = set()
        for kf_id in recent_kf_ids:
            active_pt_ids.update(self.keyframes[kf_id].observed_point_ids)
        
        descriptors = []
        point_ids = []
        for pt_id in active_pt_ids:
            descriptors.append(self.points[pt_id].descriptor)
            point_ids.append(pt_id)
        
        return np.array(descriptors), point_ids