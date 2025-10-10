import os
import csv
import cv2
import yaml
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from robot_interface import URRobotInterface
from camera_interface import RealSenseInterface
from calibration_utils import (
    create_chessboard_3d_points,
    find_chessboard_corners_bgr,
    estimate_chessboard_pose,
    matrix_to_R_t,
    handeye_calibration
)

class FlowStyleDumper(yaml.SafeDumper):
    """ Custom YAML dumper to format matrices using flow-style lists (bracketed format) """
    def increase_indent(self, flow=False, indentless=False):
        return super(FlowStyleDumper, self).increase_indent(flow=True, indentless=False)

def represent_list_flow(dumper, data):
    """ Force flow-style representation for lists """
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

FlowStyleDumper.add_representer(list, represent_list_flow)

def save_calibration_yaml(yaml_path,
                          camera_matrix, dist_coeffs,
                          cam2ee_4x4, ee2cam_4x4):
    """Saves camera intrinsics & extrinsics to a YAML in matrix form."""
    # Ensure distortion coefficients are always 2D
    if dist_coeffs.ndim == 1:
        dist_coeffs = dist_coeffs.reshape(1, -1)

    data_dict = {
        'camera_matrix': {
            'rows': camera_matrix.shape[0],
            'cols': camera_matrix.shape[1],
            'data': camera_matrix.tolist()
        },
        'dist_coeffs': {
            'rows': dist_coeffs.shape[0],
            'cols': dist_coeffs.shape[1],
            'data': dist_coeffs.tolist()
        },
        'cam2ee': {
            'rows': 4,
            'cols': 4,
            'data': cam2ee_4x4.tolist()
        },
        'ee2cam': {
            'rows': 4,
            'cols': 4,
            'data': ee2cam_4x4.tolist()
        }
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(data_dict, f, Dumper=FlowStyleDumper, sort_keys=False, default_flow_style=False)


def main():
    # ========== Basic settings ==========
    N = 10  # Number of captures
    output_folder = "captures_1280_720"
    os.makedirs(output_folder, exist_ok=True)

    pose_csv_path = os.path.join(output_folder, "poses.csv")
    pose_csv_file = open(pose_csv_path, mode="w", newline="")
    pose_writer = csv.writer(pose_csv_file)
    pose_writer.writerow(["capture_index"] +
                         [f"base_T_ee({r},{c})" for r in range(4) for c in range(4)])

    # 1) Robot & camera init
    robot = URRobotInterface("192.168.1.254")
    camera = RealSenseInterface(width=1280, height=720, fps=30)
    cam_matrix = camera.camera_matrix
    dist_coeffs = camera.dist_coeffs

    # Force dist_coeffs to 2D if needed
    if dist_coeffs.ndim == 1:
        dist_coeffs = dist_coeffs.reshape(1, -1)

    print("Camera Intrinsics:\n", cam_matrix)
    print("Distortion Coeffs:", dist_coeffs)

    # 2) Checkerboard config
    pattern_size = (12, 9)
    square_size = 0.015
    board_points_3d = create_chessboard_3d_points(pattern_size, square_size)

    # For calibrateHandEye, we store 'R_gripper2base' and 'R_target2cam'
    # but we must keep them consistent with the "from B->A" convention that cv2 wants:
    #   * R_gripper2base: ^base T_gripper
    #   * R_target2cam  : ^camera T_target
    # If we treat "ee" as "gripper" and "board" as "target," it's consistent.
    R_gripper2base_all = []
    t_gripper2base_all = []
    R_target2cam_all = []
    t_target2cam_all = []

    image_count = 0

    for i in range(N):
        print(f"\n=== Capture #{i + 1}/{N} ===")
        print("Position the robot & checkerboard so corners are detected clearly.")
        print("Press [ESC] in the OpenCV window to capture this pose.")

        # 2A) Detect corners
        corners_2d = None
        while True:
            raw_image = camera.get_color_frame()
            display = raw_image.copy()
            found, corners_temp = find_chessboard_corners_bgr(display, pattern_size, draw=True)
            if found:
                corners_2d = corners_temp
            cv2.putText(display, "Press ESC to capture", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("RealSense Stream", display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

        if corners_2d is None:
            print("No corners detected. Skipping.")
            continue

        # 2B) Robot pose: suppose robot.get_pose_matrix() returns a 4x4 named T_base_ee = ^base T_ee
        T_base_ee = robot.get_pose_matrix()
        base_T_ee_flat = T_base_ee.flatten().tolist()

        # calibrateHandEye wants 'gripper->base' = ^base T_gripper
        # i.e. from gripper to base. But ^base T_ee is "from ee to base."
        # That is exactly the same meaning as "gripper->base."
        # so T_base_ee is ^base T_ee => that is R_gripper2base if we treat ee=gripper
        # No inversion needed if your function indeed returns from ee to base.
        # But some UR APIs return the opposite. Double-check!
        R_base_ee = T_base_ee[:3, :3]
        t_base_ee = T_base_ee[:3, 3]

        image_count += 1
        img_filename = f"capture_{image_count}.png"
        img_path = os.path.join(output_folder, img_filename)
        cv2.imwrite(img_path, raw_image)
        print(f"Saved raw image to {img_path}")

        row = [image_count] + base_T_ee_flat
        pose_writer.writerow(row)
        pose_csv_file.flush()

        # 2C) Chessboard pose in camera: estimate_chessboard_pose => ^camera T_board
        T_camera_board = estimate_chessboard_pose(corners_2d, board_points_3d,
                                                  cam_matrix, dist_coeffs)
        if T_camera_board is None:
            print("solvePnP failed. Skipping.")
            continue

        # For calibrateHandEye => R_target2cam means ^camera T_target
        # i.e. from target(board) to camera. Our function returns exactly that.
        R_camera_board = T_camera_board[:3, :3]
        t_camera_board = T_camera_board[:3, 3]

        # 2D) Store them
        # We call ^base T_ee the "gripper->base" => R_gripper2base, t_gripper2base
        # because calibrateHandEye doc says:
        #   'R_gripper2base': array of rotation from gripper to base
        #   'R_target2cam':   array of rotation from target to cam
        R_gripper2base_all.append(R_base_ee)
        t_gripper2base_all.append(t_base_ee)

        R_target2cam_all.append(R_camera_board)
        t_target2cam_all.append(t_camera_board)

    cv2.destroyAllWindows()

    # 3) Perform calibrateHandEye
    print("Running calibrateHandEye...")
    # We can pick any method: e.g. cv2.CALIB_HAND_EYE_TSAI
    R_cam2ee, t_cam2ee = cv2.calibrateHandEye(
        R_gripper2base_all, t_gripper2base_all,
        R_target2cam_all, t_target2cam_all,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    # R_cam2ee = ^ee R_camera
    # So we build a 4x4 'cam2ee' = ^ee T_camera => from camera to ee
    # But note that calibrateHandEye doc states it returns cam2gripper, i.e. ^gripper T_cam.
    # If we interpret 'ee' as 'gripper,' that is consistent.
    # => ^gripper R_cam2gripper
    # Let's store it as T_ee_camera => from camera to ee:
    T_ee_camera = np.eye(4)
    T_ee_camera[:3, :3] = R_cam2ee
    T_ee_camera[:3, 3] = t_cam2ee.ravel()

    # If you prefer naming it 'cam2ee' in your code, do that, just ensure you know the direction.
    print("calibrateHandEye result => ^ee T_camera (cam2ee):\n", T_ee_camera)

    # 4) Invert if you want 'camera->ee' or 'ee->camera' the other way
    T_camera_ee = np.linalg.inv(T_ee_camera)
    print("^camera T_ee (the inverse):\n", T_camera_ee)



    # 6) Save everything to a final YAML
    output_yaml = os.path.join(output_folder, "final_calibration.yaml")
    save_calibration_yaml(
        yaml_path=output_yaml,
        camera_matrix=cam_matrix,
        dist_coeffs=dist_coeffs,
        cam2ee_4x4=T_camera_ee,
        ee2cam_4x4=T_ee_camera
    )
    print(f"\nSaved calibration (intrinsics + extrinsics) to {output_yaml}.")


    cv2.destroyAllWindows()
    camera.stop()
    robot.disconnect()
    pose_csv_file.close()

if __name__ == "__main__":
    main()
