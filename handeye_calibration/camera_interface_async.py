# camera_interface.py

import pyrealsense2 as rs
import numpy as np
import cv2
import threading
import yaml
from pathlib import Path

class RealSenseInterfaceAsync:
    def __init__(self, calibration_path, out_dir, width=1280, height=720, fps=30):
        self.running = False
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)

        colorizer = rs.colorizer()
        colorizer.set_option(rs.option.visual_preset, 1) # 0=Dynamic, 1=Fixed, 2=Near, 3=Far
        colorizer.set_option(rs.option.min_distance, 0.0)
        colorizer.set_option(rs.option.max_distance, 1.0)
        self.colorizer = colorizer

        # 🔹 Apply camera settings for best quality
        self.device = self.profile.get_device()
        self.sensors = self.device.query_sensors()
        self.depth_scale = self.device.first_depth_sensor().get_depth_scale()

        frame_size = (1280, 720)
        fps = 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        filename = out_dir / "recording.mov"
        print(f"Saving recording to {str(filename)}")
        self.writer = cv2.VideoWriter(str(filename),fourcc,fps,frame_size)

        for sensor in self.sensors:
            if sensor.is_depth_sensor():
                # print("Configuring depth sensor")
                sensor.set_option(rs.option.visual_preset, rs.rs400_visual_preset.high_accuracy)
                # sensor.set_option(rs.option.visual_preset, 5)   # High-accuracy preset
                sensor.set_option(rs.option.laser_power, 360)   # Max laser power
                sensor.set_option(rs.option.emitter_enabled, 1) # Enable depth emitter

            else:  # RGB Sensor
                continue
                # sensor.set_option(rs.option.sharpness, 100)  # Increase sharpness
                # sensor.set_option(rs.option.exposure, 100)  # Adjust exposure (manual mode)
                # sensor.set_option(rs.option.gain, 16)  # Increase gain for better brightness
                # sensor.set_option(rs.option.white_balance, 4500)  # Adjust white balance
                # sensor.set_option(rs.option.enable_auto_exposure, 1)  # Enable auto exposure


        # Retrieve camera intrinsics from RealSense
        intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.camera_matrix = np.array([[intr.fx, 0, intr.ppx],
                                       [0, intr.fy, intr.ppy],
                                       [0, 0, 1]], dtype=np.float64)
        self.distortion_coeffs = np.array(intr.coeffs, dtype=np.float64)  # [k1, k2, p1, p2, k3] etc.

        # Load configuration
        with open(calibration_path, "r") as file:
            calibration = yaml.safe_load(file)
        self.T_cam_matrix = np.array(calibration["ee2cam"]["data"])

        # 🔹 Multi-threaded frame processing to improve FPS
        self.color_image = np.array([])
        self.depth_image = np.array([])
        self.depth_colormap = np.array([])
        self.running = True
        self.thread = threading.Thread(target=self.update_frames, daemon=True)
        self.thread.start()

        # Make sure we're getting frames
        while True:
            c,z,zm = self.get_frames()
            # print(f"{c.size} {z.size} {zm.size}")
            if c.size and z.size and zm.size: break

    def update_frames(self):
        """Continuously update frames in a separate thread for real-time processing"""

        while self.running:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame().as_depth_frame()
            self.color_image = np.asanyarray(color_frame.get_data())
            self.depth_image = np.asanyarray(depth_frame.get_data())
            self.depth_colormap = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())

            # self.writer.write(self.color_image)

            cv2.imshow("Camera", self.color_image)
            cv2.waitKey(1)



    def get_frames(self):
        """Retrieve the latest color and depth frames"""
        if self.color_image.size and self.depth_image.size and self.depth_colormap.size:
            color_image = self.color_image
            depth_image = self.depth_image
            depth_colormap = self.depth_colormap

            # 🔹 Apply post-processing filters to improve depth quality
            # depth_frame_filtered = self.decimation.process(self.depth_frame)
            # depth_frame_filtered = self.spatial.process(depth_frame_filtered)
            # depth_frame_filtered = self.temporal.process(depth_frame_filtered)

            # depth_filtered_image = np.asanyarray(depth_frame_filtered.get_data())

            # Conver to meters
            depth_image = depth_image.astype(np.float32) * self.depth_scale

            # For visualization
            # Qui moltiplichiamo per 255 se vogliamo visualizzare come immagine (questa parte non altera i dati salvati)
            # depth_colormap = cv2.applyColorMap(
            #     cv2.convertScaleAbs(depth_image, alpha=255/depth_image.max()),
            #     cv2.COLORMAP_JET
            # )

            return color_image, depth_image, depth_colormap

        return np.array([]), np.array([]), np.array([])

    def stop(self):
        self.__del__()

    def __del__(self):
        """Stop RealSense and terminate threading"""
        print("Disconnecting from camera")
        if self.running:
            self.running = False
            self.thread.join()
            print("Stopping camera")
            self.writer.release()
            self.pipeline.stop()
