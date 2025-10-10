# camera_interface.py

import pyrealsense2 as rs
import numpy as np
import cv2


class RealSenseInterface:
    def __init__(self, width=1280, height=720, fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.profile = self.pipeline.start(self.config)

        # Retrieve camera intrinsics from RealSense
        intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.camera_matrix = np.array([[intr.fx, 0, intr.ppx],
                                       [0, intr.fy, intr.ppy],
                                       [0, 0, 1]], dtype=np.float64)
        self.dist_coeffs = np.array(intr.coeffs, dtype=np.float64)  # [k1, k2, p1, p2, k3] etc.

    def get_color_frame(self):
        """
        Grab the latest color frame as a numpy BGR image.
        """
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        return color_image

    def stop(self):
        self.pipeline.stop()
