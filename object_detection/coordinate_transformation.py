import numpy as np
import yaml
import cv2

def load_camera_params(config_path="configs/config_720p.yaml"):
    """Load camera calibration parameters from YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    intrinsic_matrix = np.array(config["camera"]["intrinsic_matrix"])
    distortion_coeffs = np.array(config["camera"]["distortion_coeffs"])

    rotation_matrix = np.array(config["extrinsic"]["rotation_matrix"])
    translation_vector = np.array(config["extrinsic"]["translation_vector"]).reshape(3, 1)

    # Construct End-Effector to Camera Transformation (T_cam)
    T_cam_matrix = np.eye(4)
    T_cam_matrix[:3, :3] = rotation_matrix
    T_cam_matrix[:3, 3] = translation_vector.flatten()

    return intrinsic_matrix, distortion_coeffs, T_cam_matrix

def pixel_to_camera(pixel_coords, depth, intrinsic_matrix):
    """Convert 2D pixel coordinates + depth to 3D camera frame coordinates."""
    u, v = pixel_coords
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    # Compute 3D coordinates in the camera frame
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth

    return np.array([X, Y, Z, 1])  # Homogeneous coordinates

def transform_to_robot_frame(pixel_coords, depth, baseTee_matrix, intrinsic_matrix, T_cam_matrix):
    """Convert pixel coordinates to robot base frame using depth data."""

    # Convert to camera coordinates
    cam_coords = pixel_to_camera(pixel_coords, depth, intrinsic_matrix)

    # Transform to end-effector frame
    ee_coords = np.dot(T_cam_matrix, cam_coords)

    # Transform to robot base frame using real-time baseTee
    base_coords = np.dot(baseTee_matrix, ee_coords)

    return base_coords[:3], cam_coords[:3], ee_coords[:3]  # Return only X, Y, Z (ignore homogeneous component)
