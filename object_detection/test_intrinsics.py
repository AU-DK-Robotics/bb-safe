import pyrealsense2 as rs
import numpy as np


def get_intrinsics(resolution=(640, 480), fps=30):
    """
    Retrieve the intrinsic parameters of the RealSense camera for a given resolution.

    :param resolution: Tuple (width, height) for the desired resolution.
    :param fps: Frames per second (default: 30).
    :return: Intrinsic matrix and distortion coefficients.
    """
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable stream with the specified resolution
    width, height = resolution
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    # Start pipeline to get the active profile
    pipeline_profile = pipeline.start(config)

    # Get color sensor intrinsics
    color_stream = pipeline_profile.get_stream(rs.stream.color)
    intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

    # Stop pipeline after retrieving intrinsics
    pipeline.stop()

    # Build intrinsic matrix
    intrinsic_matrix = np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ])

    # Distortion coefficients
    distortion_coeffs = np.array(intrinsics.coeffs)

    return intrinsic_matrix, distortion_coeffs, intrinsics


# Test different resolutions
resolutions = [(640, 480), (1280, 720), (1920, 1080)]

for res in resolutions:
    print(f"\n📌 Resolution: {res}")
    intrinsics, distortion, intrinsics_obj = get_intrinsics(res)

    print("Intrinsic Matrix:")
    print(intrinsics)

    print("Distortion Coefficients:")
    print(distortion)

    print(f"Focal Length (fx, fy): ({intrinsics_obj.fx}, {intrinsics_obj.fy})")
    print(f"Principal Point (cx, cy): ({intrinsics_obj.ppx}, {intrinsics_obj.ppy})")
    print(f"Distortion Model: {intrinsics_obj.model}")
