import torch
import yaml
import cv2
import numpy as np
import pyrealsense2 as rs
import threading
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self,calibration_path,model_weights_path,confidence_threshold):
        """Initialize YOLO model and optimized RealSense camera"""

        self.confidence_threshold = confidence_threshold

        # Load configuration
        with open(calibration_path, "r") as file:
            calibration = yaml.safe_load(file)

        # Load extrinsics from config
        self.T_cam_matrix = np.array(calibration["ee2cam"]["data"])

        # Load YOLOv8 model
        self.model = YOLO(model_weights_path)

        # Initialize RealSense camera
        self.pipeline = rs.pipeline()
        config = rs.config()

        # 🔹 Enable high-resolution streams with higher FPS
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)  # Higher FPS
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

        # Start pipeline
        self.profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        self.depth_scale = self.get_depth_scale()

        # 🔹 Apply camera settings for best quality
        self.set_camera_settings()

        # Retrieve camera intrinsics from RealSense
        intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.camera_matrix = np.array([[intr.fx, 0, intr.ppx],
                                       [0, intr.fy, intr.ppy],
                                       [0, 0, 1]], dtype=np.float64)
        self.distortion_coeffs = np.array(intr.coeffs, dtype=np.float64)  # [k1, k2, p1, p2, k3] etc.

        # 🔹 Initialize post-processing filters (for depth quality)
        # self.decimation = rs.decimation_filter()  # Reduces depth resolution for better performance
        # self.spatial = rs.spatial_filter()  # Smooths depth values (removes noise)
        # self.temporal = rs.temporal_filter()  # Reduces flickering noise

        # 🔹 Multi-threaded frame processing to improve FPS
        self.color_frame = None
        self.depth_frame = None
        self.running = True
        self.thread = threading.Thread(target=self.update_frames, daemon=True)
        self.thread.start()

    def set_camera_settings(self):
        """Adjust RealSense settings for best image and depth quality"""
        device = self.profile.get_device()
        sensors = device.query_sensors()

        for sensor in sensors:
            if sensor.is_depth_sensor():
                sensor.set_option(rs.option.visual_preset, 5)  # High-accuracy preset
                sensor.set_option(rs.option.laser_power, 100)  # Max laser power
                sensor.set_option(rs.option.emitter_enabled, 1)  # Enable depth emitter

            else:  # RGB Sensor
                None
                # sensor.set_option(rs.option.sharpness, 100)  # Increase sharpness
                # sensor.set_option(rs.option.exposure, 100)  # Adjust exposure (manual mode)
                # sensor.set_option(rs.option.gain, 16)  # Increase gain for better brightness
                # sensor.set_option(rs.option.white_balance, 4500)  # Adjust white balance
                # sensor.set_option(rs.option.enable_auto_exposure, 1)  # Enable auto exposure


        print("✅ RealSense camera settings applied.")

    def get_depth_scale(self):
        """Retrieve the depth scale from the RealSense sensor"""
        depth_sensor = self.profile.get_device().first_depth_sensor()
        return depth_sensor.get_depth_scale()

    def update_frames(self):
        """Continuously update frames in a separate thread for real-time processing"""
        while self.running:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)

            self.color_frame = aligned_frames.get_color_frame()
            self.depth_frame = aligned_frames.get_depth_frame()

    def get_frames(self):
        """Retrieve the latest color and depth frames"""
        if self.color_frame and self.depth_frame:
            color_image = np.asanyarray(self.color_frame.get_data())
            depth_image = np.asanyarray(self.depth_frame.get_data())

            # 🔹 Apply post-processing filters to improve depth quality
            # depth_frame_filtered = self.decimation.process(self.depth_frame)
            # depth_frame_filtered = self.spatial.process(depth_frame_filtered)
            # depth_frame_filtered = self.temporal.process(depth_frame_filtered)

            # depth_filtered_image = np.asanyarray(depth_frame_filtered.get_data())

            # Conver to meters
            depth_image = depth_image.astype(np.float32) * self.depth_scale

            # For visualization
            # Qui moltiplichiamo per 255 se vogliamo visualizzare come immagine (questa parte non altera i dati salvati)
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=255/depth_image.max()),
                cv2.COLORMAP_JET
            )

            return color_image, depth_image, depth_colormap

        return None, None, None

    def detect_objects(self, image, depth_frame):
        """Detect objects and return bounding boxes with real-world coordinates"""
        results = self.model(image)[0]

        height, width = depth_frame.shape  # Get depth image size

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])

            if confidence > self.confidence_threshold:
                # Compute center of bounding box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # 🔹 Ensure center_x, center_y are within valid depth image range
                center_x = max(0, min(center_x, width - 1))
                center_y = max(0, min(center_y, height - 1))

                # Retrieve depth value safely
                depth_spot =  (center_x, center_y)
                depth_value = depth_frame[depth_spot]
                # depth_value = depth_frame[346, 509]

                detections.append({
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "confidence": confidence,
                    "class_id": class_id,
                    "depth": depth_value,
                    "center": (center_x, center_y),
                    "depth_spot": depth_spot
                })

        return detections

    def stop(self):
        """Stop RealSense and terminate threading"""
        self.running = False
        self.thread.join()
        self.pipeline.stop()
