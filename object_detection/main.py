import cv2
import numpy as np
import csv
from old_detector import ObjectDetector
from coordinate_transformation import transform_to_robot_frame
from rtde_receive import RTDEReceiveInterface

config_path = "configs/config_720p.yaml"
od_modelweights_path = "runs/detect/train2/weights/best.pt"

# Initialize RealSense-enabled YOLO detector
detector = ObjectDetector(config_path,od_modelweights_path)

# Connect to UR robot for real-time pose
robot_ip = "192.168.1.254"
rtde_receive = RTDEReceiveInterface(robot_ip)

# Initialize a list to store object detections
object_data = []

# Open a CSV file to log data (write mode, with header)
csv_file = "detections_log.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Class ID", "Confidence", "X", "Y", "Z"])  # Header

def get_pose_matrix():
    """
    Return ^base T_ee as a 4x4 homogeneous transformation matrix.
    UR 'getActualTCPPose()' returns [x, y, z, Rx, Ry, Rz] in meters / axis-angle.
    """
    tcp_pose = rtde_receive.getActualTCPPose()
    x, y, z, rx, ry, rz = tcp_pose

    # Convert axis-angle (rotation vector) to rotation matrix
    rotation_vector = np.array([rx, ry, rz], dtype=float)
    R, _ = cv2.Rodrigues(rotation_vector)  # Convert to 3x3 rotation matrix

    # Build the 4x4 homogeneous transformation matrix
    pose_mat = np.eye(4)
    pose_mat[:3, :3] = R
    pose_mat[:3, 3] = [x, y, z]

    return pose_mat


while True:
    # Get RGB and depth frames
    color_frame, depth_frame = detector.get_frames()
    if color_frame is None or depth_frame is None:
        continue  # Skip frame if retrieval fails

    # Detect objects
    detections = detector.detect_objects(color_frame, depth_frame)

    # Get real-time baseTee matrix from URTDE
    baseTee_matrix = get_pose_matrix()

    for detection in detections:
        x, y, w, h = detection["bbox"]
        confidence = detection["confidence"]
        class_id = detection["class_id"]
        depth = detection["depth"]
        center_x, center_y = detection["center"]

        # Compute real-world coordinates
        object_coords = transform_to_robot_frame((center_x, center_y), depth, baseTee_matrix)
        obj_x, obj_y, obj_z = object_coords  # Extract X, Y, Z

        # Save detection to list
        object_data.append([class_id, confidence, obj_x, obj_y, obj_z])

        # Draw bounding box and coordinates
        cv2.rectangle(color_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(color_frame, f"Class: {class_id} | ({obj_x:.3f}, {obj_y:.3f}, {obj_z:.3f})",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save data to CSV after processing each frame
    # with open(csv_file, mode='a', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(object_data)

    # Clear list after saving
    # object_data.clear()

    # Display the results
    cv2.imshow("RealSense YOLO Detection", color_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
