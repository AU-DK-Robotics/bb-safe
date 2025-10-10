# trajectory_planner.py
import numpy as np


def linear_trajectory(start_pose, goal_pose, num_points):
    """
    Generate a linear trajectory from start_pose to goal_pose.

    Parameters:
      start_pose: (numpy array) 6D pose (x, y, z, rx, ry, rz) at the start.
      goal_pose: (numpy array) 6D pose at the goal.
      num_points: (int) number of trajectory points.

    Returns:
      A list of numpy arrays representing the trajectory.
    """
    trajectory = []
    for i in range(num_points):
        ratio = i / (num_points - 1)
        point = start_pose + (goal_pose - start_pose) * ratio
        trajectory.append(point)
    return trajectory
