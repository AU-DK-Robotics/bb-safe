# camera_interface.py

import pyrealsense2 as rs
import numpy as np
import cv2
import threading
import yaml
from pathlib import Path
from camera_utils import snow
import time
class RealSenseInterfaceAsync:
    def __init__(self, width=1280, height=720, fps=30, hec_path=None, recording_path=None, snow_factor=0.0, snow_rate=0.0):

        # Reset all connected Realsense devices
        ctx = rs.context()
        devices = ctx.query_devices()
        for dev in devices:
            dev.hardware_reset()
        time.sleep(5)

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
        self.depth_sensor = self.device.first_depth_sensor()
        self.color_sensor = self.device.first_color_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()

        # self.depth_sensor.set_option(rs.option.visual_preset, rs.rs400_visual_preset.high_accuracy)
        # sensor.set_option(rs.option.visual_preset, 5)   # High-accuracy preset
        # self.depth_sensor.set_option(rs.option.laser_power, 360)   # Max laser power
        # self.depth_sensor.set_option(rs.option.emitter_enabled, 1) # Enable depth emitter

        self.recording_path = recording_path
        if self.recording_path:
            frame_size = (width, height)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            filename = recording_path / "recording.mov"
            print(f"Saving recording to {str(filename)}")
            self.writer = cv2.VideoWriter(str(filename),fourcc,fps,frame_size)

        # Retrieve camera intrinsics from RealSense
        intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.camera_matrix = np.array([[intr.fx, 0, intr.ppx],
                                       [0, intr.fy, intr.ppy],
                                       [0, 0, 1]], dtype=np.float64)
        self.distortion_coeffs = np.array(intr.coeffs, dtype=np.float64)  # [k1, k2, p1, p2, k3] etc.

        # Load HEC configuration
        if hec_path:
            with open(hec_path, "r") as file:
                calibration = yaml.safe_load(file)
            self.T_cam_matrix = np.array(calibration["ee2cam"]["data"])

        # 🔹 Multi-threaded frame processing to improve FPS
        self.color_image = np.array([])
        self.depth_image = np.array([])
        self.depth_colormap = np.array([])
        self.running = True
        self.thread = threading.Thread(target=self.update_frames, daemon=True)
        self.thread.start()

        self.snow_factor = snow_factor
        self.snow_rate = snow_rate

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

            # Apply snow to RGB image
            if self.snow_factor > 0:
                snow_mean,snow_std=self.snow_model()
                self.color_image, _ = snow.apply(self.color_image,mean=snow_mean,std=snow_std)

            if self.recording_path:
                self.writer.write(self.color_image)

            cv2.imshow("Camera", self.color_image)
            cv2.waitKey(1)

    def snow_model(self):
        # factor: sensitivity conversion from Picam R2 to RealSense D435
        # snow_rate: gamma radiation variable for linear regression

        rs_gain = self.color_sensor.get_option(rs.option.gain)
        rs_texp = self.color_sensor.get_option(rs.option.exposure)/10000 # convert units of 100 microsec -> 1 second

        factor = self.snow_factor*rs_gain*rs_texp
        print(f"Final factor: {factor} (Gain: {rs_gain}, Exposure: {rs_texp} sec)")
        mean_0 = 34.01
        mean_slope = 42.11
        mean = factor*(mean_0 + mean_slope*np.sqrt(self.snow_rate))
        var_0 = 0.81
        var_slope = 6.7
        var = factor*(var_0 + var_slope*np.sqrt(self.snow_rate))
        return mean,var

    def get_frames(self):
        """Retrieve the latest color and depth frames"""
        if self.color_image.size and self.depth_image.size and self.depth_colormap.size:

            # Get images
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
