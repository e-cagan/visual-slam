"""
Module for visualizations.
"""

import os
import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(traj, title="Ground-truth trajectory"):
    """Plots traj: (N, 3). plots X-Z plane: x axis = traj[:,0], y axis = traj[:,2]; ty (traj[:,1]) = height"""
    # Create results dir to avoid error (skip if already present)
    os.makedirs("results", exist_ok=True)
    
    # Plot info
    plt.title(title)
    plt.plot(traj[:, 0], traj[:, 2])
    plt.axis('equal')
    plt.xlabel("X")
    plt.ylabel("Z")
    
    # Plot saving
    plt.savefig("results/trajectory.png")
    print("Trajectory plot has been saved successfully to 'results/trajectory.png'.")
    plt.show()