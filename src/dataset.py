"""
Module for kitti dataset.
"""

import os
import glob
import numpy as np
import cv2


class KittiDataset:
    """
    Kitti dataset class.
    """
    
    def __init__(self, sequence_id, base_path="data/dataset"):
        self.sequence_path = os.path.join(base_path, 'sequences', sequence_id)
        self.frame_paths = sorted(glob.glob(os.path.join(self.sequence_path, "image_2", "*.png")))
        self.frame_paths_right = sorted(glob.glob(os.path.join(self.sequence_path, "image_3", "*.png")))
        self.P_left = self._parse_proj_matrix(os.path.join(f'{self.sequence_path}', 'calib.txt'), label="P2")    # 3x4
        self.P_right = self._parse_proj_matrix(os.path.join(f'{self.sequence_path}', 'calib.txt'), label="P3")   # 3x4
        self.K = self.P_left[:, :3] if self.P_left is not None else None
        self.poses = self._parse_poses(os.path.join(base_path, "poses", f"{sequence_id}.txt"))
    
    def _parse_proj_matrix(self, path, label):
        """Generalized calib parser. Returns: (3, 4) ndarray"""
        # Check if the path exists
        if not os.path.exists(path): 
            return None

        with open(path, "r") as file:
            for line in file:
                if line.startswith(f"{label}:"):
                    data_part = line.split(f"{label}:")[1]
                    values = [float(x) for x in data_part.strip().split()]
                    
                    return np.array(values).reshape(3, 4)
                
        raise ValueError(f"Couldn't find {label} row in the calibration file: {path}")

    def _parse_poses(self, path):
        """Every row → 12 float → reshape(3,4) → (N,3,4)"""
        # Check if the path exists
        if not os.path.exists(path): 
            return None
        
        poses = []

        with open(path, "r") as file:
            for line in file:
                if not line.strip(): 
                    continue
                # Convert strings to floats
                values = [float(x) for x in line.split()]
                poses.append(values)

        if not poses:
            raise ValueError(f"Couldn't fine any pose on the poses file: {path}")
        
        # Create the poses numpy matrix with reshape (-1 for autocalculation for that axis)
        poses_matrix = np.array(poses).reshape(-1, 3, 4)
        
        return poses_matrix

    def __len__(self):
        """Returns the length of frame paths."""
        return len(self.frame_paths)

    def get_trajectory(self):
        """Take every pose's last column (t) from self.poses → (N, 3)"""
        if self.poses is None:
            return None
        
        return self.poses[:, :, 3]
    
    def get_stereo_frame(self, i):
        """Returns: (img_left, img_right) — both of them are grayscale ndarray"""
        img_left = cv2.imread(self.frame_paths[i], cv2.IMREAD_GRAYSCALE)
        img_right = cv2.imread(self.frame_paths_right[i], cv2.IMREAD_GRAYSCALE)
        
        return img_left, img_right