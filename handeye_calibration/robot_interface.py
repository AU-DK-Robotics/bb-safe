# robot_interface.py

import numpy as np
import cv2
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface


class URRobotInterface:
    def __init__(self, robot_ip="192.168.1.254"):
        self.robot_ip = robot_ip
        # self.rtde_c = RTDEControlInterface(self.robot_ip)
        self.rtde_r = RTDEReceiveInterface(self.robot_ip)

    def get_pose_matrix(self):
        """
        Return ^base T_ee as a 4x4 homogeneous matrix.
        UR 'getActualTCPPose()' returns [x, y, z, Rx, Ry, Rz] in meters / axis-angle.
        """
        tcp_pose = self.rtde_r.getActualTCPPose()
        x, y, z, rx, ry, rz = tcp_pose

        # Convert axis-angle (rotation vector) -> rotation matrix
        rotation_vector = np.array([rx, ry, rz], dtype=float)
        R, _ = cv2.Rodrigues(rotation_vector)  # Correctly converts to 3x3 rotation matrix

        # Build the 4x4 homogeneous transformation matrix
        pose_mat = np.eye(4)
        pose_mat[:3, :3] = R
        pose_mat[:3, 3] = [x, y, z]

        return pose_mat

    def disconnect(self):
        # self.rtde_c.disconnect()
        self.rtde_r.disconnect()
