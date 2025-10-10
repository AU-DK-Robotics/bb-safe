import os
import csv
import numpy as np
import cv2
import time

from robot_interface import URRobotInterface
from camera_interface import RealSenseInterface
from calibration_utils import (
    find_chessboard_corners_bgr,
    create_chessboard_3d_points,
    estimate_chessboard_pose,
    matrix_to_R_t,
    handeye_calibration
)

def main():
    # 1) Initialize Robot
    robot = URRobotInterface(robot_ip="192.168.1.254")  # Adjust IP as needed

    # 2) Initialize RealSense
    camera = RealSenseInterface(width=640, height=480, fps=30)
    cam_matrix = camera.camera_matrix
    dist_coeffs = camera.dist_coeffs
    print("Camera Intrinsics:\n", cam_matrix)
    print("Distortion Coeffs:", dist_coeffs)

    # 3) Chessboard configuration
    pattern_size = (12, 9)   # (number_of_corners_along_width, number_of_corners_along_height)
    square_size = 0.015      # meters per square
    object_points_3d = create_chessboard_3d_points(pattern_size, square_size)

    # We'll collect these for calibrateHandEye
    R_gripper2base_all = []
    t_gripper2base_all = []
    R_target2cam_all   = []
    t_target2cam_all   = []

    # Make a folder to store captured images
    output_folder = "captures"
    os.makedirs(output_folder, exist_ok=True)

    # Create or open a CSV to store poses
    pose_csv_path = os.path.join(output_folder, "poses.csv")
    csv_file = open(pose_csv_path, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    # Write CSV header (optional)
    csv_writer.writerow(["capture_index",
                         "base_T_ee(0,0)", "base_T_ee(0,1)", "base_T_ee(0,2)", "base_T_ee(0,3)",
                         "base_T_ee(1,0)", "base_T_ee(1,1)", "base_T_ee(1,2)", "base_T_ee(1,3)",
                         "base_T_ee(2,0)", "base_T_ee(2,1)", "base_T_ee(2,2)", "base_T_ee(2,3)"])

    # Number of calibration captures
    N = 10

    try:
        for i in range(N):
            print(f"\n=== Capture #{i+1}/{N} ===")
            print("Position the robot & chessboard so the camera sees it clearly.")
            print("Press [ESC] in the OpenCV window to finalize the capture for this pose.")

            corners_2d = None
            success = False

            while True:
                # Stream frames
                color_image = camera.get_color_frame()

                # Attempt to find chessboard
                success_temp, corners_2d_temp = find_chessboard_corners_bgr(
                    color_image,
                    pattern_size,
                    draw=True
                )
                if success_temp:
                    corners_2d = corners_2d_temp
                    # Draw the corners for visualization
                    cv2.drawChessboardCorners(color_image, pattern_size, corners_2d, True)

                cv2.putText(color_image, "Press ESC to capture this frame", (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.imshow("RealSense Stream", color_image)

                key = cv2.waitKey(1) & 0xFF
                # If ESC, finalize capture
                if key == 27:
                    success = success_temp  # store last detection flag
                    break

            # We have either found corners or not
            if not success:
                print("No valid chessboard corners found in this capture! Retry or move the board.")
                continue

            # 3A) Get robot pose ^base T_ee
            base_T_ee = robot.get_pose_matrix()
            ee_T_base = np.linalg.inv(base_T_ee)

            # 3B) Save the color image
            img_filename = f"capture_{i+1}.png"
            img_path = os.path.join(output_folder, img_filename)
            cv2.imwrite(img_path, color_image)
            print(f"Saved image to {img_path}")

            # 3C) Save the robot pose to CSV
            flat_pose = base_T_ee[:3,:].flatten()  # e.g. 12 values (3x4)
            row = [i+1] + list(flat_pose)
            csv_writer.writerow(row)
            csv_file.flush()

            # 3D) Estimate chessboard pose ^camera T_board
            camera_T_board = estimate_chessboard_pose(
                corners_2d,
                object_points_3d,
                cam_matrix,
                dist_coeffs
            )
            if camera_T_board is None:
                print("solvePnP failed to estimate pose. Skipping capture.")
                continue

            # 3E) Invert to get board->camera
            board_T_camera = np.linalg.inv(camera_T_board)

            # Convert to R,t
            R_ee2base, t_ee2base = matrix_to_R_t(ee_T_base)
            R_board2cam, t_board2cam = matrix_to_R_t(board_T_camera)

            # Store for calibrateHandEye
            R_gripper2base_all.append(R_ee2base)
            t_gripper2base_all.append(t_ee2base)
            R_target2cam_all.append(R_board2cam)
            t_target2cam_all.append(t_board2cam)

        cv2.destroyAllWindows()

        # 4) Perform Hand-Eye Calibration
        print("\nPerforming Hand-Eye Calibration with OpenCV ...")
        if len(R_gripper2base_all) < 2:
            print("Not enough valid captures to run calibration.")
        else:
            R_cam2ee, t_cam2ee = handeye_calibration(
                R_gripper2base_all,
                t_gripper2base_all,
                R_target2cam_all,
                t_target2cam_all,
                method=cv2.CALIB_HAND_EYE_TSAI
            )

            # Build 4x4 transform
            cam2ee = np.eye(4)
            cam2ee[:3,:3] = R_cam2ee
            cam2ee[:3, 3] = t_cam2ee.ravel()

            print("=== Calibration Results ===")
            print("Camera->End-Effector transform:\n", cam2ee)

            ee2cam = np.linalg.inv(cam2ee)
            print("\nEnd-Effector->Camera transform:\n", ee2cam)

    finally:
        # Cleanup
        cv2.destroyAllWindows()
        camera.stop()
        robot.disconnect()
        csv_file.close()

if __name__ == "__main__":
    main()
