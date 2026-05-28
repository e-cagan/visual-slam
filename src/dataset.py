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
        self.K = self._parse_calib(os.path.join(f'{self.sequence_path}', 'calib.txt'))
        self.poses = self._parse_poses(os.path.join(base_path, "poses", f"{sequence_id}.txt"))
        
        # DEBUG
        print(len(self.frame_paths))      # for 04 -> ~270
        print(self.K.shape)               # (3,3), not string
        print(self.poses.shape)           # (N, 3, 4)

    def _parse_calib(self, path):
        """Finds P2 row → 12 float → reshape(3,4) → [:, :3] → K"""
        if not os.path.exists(path): 
            return None
        
        p0_values = []

        # Parse the calibration file
        with open(path, "r") as file:
            for line in file:
                if line.startswith("P2"):
                    data_part = line.split("P2:")[1]
                    # Convert strings to floats
                    p0_values = [float(x) for x in data_part.strip().split()]
                    break
                    
        if not p0_values:
            raise ValueError("Couldn't find P2 row in the calibration file.")
        
        # Create the numpy matrix then reshape it
        p0_matrix = np.array(p0_values).reshape(3, 4)

        # Take the K matrix within P0
        K = p0_matrix[:, :3]
        return K

    def _parse_poses(self, path):
        """Every row → 12 float → reshape(3,4) → (N,3,4)"""
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
            raise ValueError("Couldn't fine any pose on the poses file.")
        
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