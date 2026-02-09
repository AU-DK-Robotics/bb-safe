import torch
import yaml
import cv2
import numpy as np
import threading
from ultralytics import YOLO  # pyright: ignore[reportPrivateImportUsage]
from datetime import datetime
from pathlib import Path

class detectorYOLO:

    MODEL_ID = "YOLO"

    def __init__(self,model_weights_path,confidence_threshold):
        """Initialize YOLO model and optimized RealSense camera"""

        self.confidence_threshold = confidence_threshold

        # Load YOLOv8 model
        self.model = YOLO(model_weights_path)

        self.infer_time = None

    def detect_objects(self, image, depth_frame, img_save_path="", log_path=""):
        """Detect objects and return bounding boxes with real-world coordinates"""
        detections = self.model(image)[0]

        if img_save_path:
            cv2.imwrite(img_save_path, image)

        # print(detections)

        height, width = depth_frame.shape  # Get depth image size

        detected_objects = []
        for box in detections.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            # print(box)

            if confidence > self.confidence_threshold:
                # Compute center of bounding box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # 🔹 Ensure center_x, center_y are within valid depth image range
                center_x = max(0, min(center_x, width - 1))
                center_y = max(0, min(center_y, height - 1))

                # Retrieve depth value safely
                # depth_spot = (346, 509)
                depth_spot = (x1 + 125, y2 - 113)
                depth_spot_yx = (depth_spot[1], depth_spot[0])
                depth_value = depth_frame[depth_spot_yx]

                class_name = detections.names[class_id]

                detected_objects.append({
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "confidence": confidence,
                    'label': class_name,
                    "class_id": class_id,
                    "depth": depth_value,
                    "center": (center_x, center_y),
                    "depth_spot": depth_spot
                })

        return detections, detected_objects
