"""
Module for V0 running script.
"""

import sys
import os

# Main path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import KittiDataset
from src.visualization import plot_trajectory


if __name__ == '__main__':
    ds = KittiDataset(sequence_id="04", base_path="data/dataset")
    traj = ds.get_trajectory()
    plot_trajectory(traj)