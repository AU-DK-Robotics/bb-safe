# calibration_utils.py

import numpy as np
import cv2


def find_chessboard_corners_bgr(image_bgr, pattern_size=(12, 9), draw=False):
    """
    Finds chessboard corners in a BGR image.
    :param pattern_size: (cols, rows) of internal corners on the chessboard.
    :return: (success, corners_2d) where corners_2d is Nx2 array of sub-pixel corners if found.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    success, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if success:
        # Refine corners for better accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        if draw:
            cv2.drawChessboardCorners(image_bgr, pattern_size, corners_refined, success)
        return True, corners_refined
    else:
        return False, None


def create_chessboard_3d_points(pattern_size=(12, 9), square_size=0.015):
    """
    Creates an array of 3D points for the specified chessboard size.
    pattern_size = (cols, rows) of internal corners
    square_size in meters.
    Return Nx3 array of object points.
    """
    # Typically we define the origin at one corner,
    # with x increasing along columns and y along rows
    # e.g. pattern_size=(7,5) means 7 corners horizontally, 5 corners vertically
    cols, rows = pattern_size
    obj_points = []
    for r in range(rows):
        for c in range(cols):
            x = c * square_size
            y = r * square_size
            z = 0.0
            obj_points.append([x, y, z])
    return np.array(obj_points, dtype=np.float32)


def estimate_chessboard_pose(corners_2d, obj_points_3d, camera_matrix, dist_coeffs):
    """
    Estimate the checkerboard pose as ^camera T_board (i.e. from 'board' to 'camera').
    That is, for a 3D point p in 'board' frame, p_cam = (^camera T_board) * p_board.
    """
    # corners_2d: shape (N, 2) or (N, 1, 2)
    # obj_points_3d: shape (N, 3), representing corners in 'board' frame

    # reshape corners if needed
    if corners_2d.ndim == 2 and corners_2d.shape[1] == 2:
        corners_2d_reshaped = corners_2d.reshape(-1, 1, 2)
    else:
        corners_2d_reshaped = corners_2d

    # SolvePnP returns rvec/tvec that map p_board -> p_cam
    success, rvec, tvec = cv2.solvePnP(
        obj_points_3d,
        corners_2d_reshaped,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    # Convert axis-angle to rotation matrix
    R_board2cam, _ = cv2.Rodrigues(rvec)

    # Build 4x4 transform: ^camera T_board = from 'board' to 'camera'
    # That means p_cam = R_board2cam * p_board + tvec
    # so for ^A T_B = from B->A, we want ^camera T_board
    T_camera_board = np.eye(4)
    T_camera_board[:3, :3] = R_board2cam
    T_camera_board[:3, 3]  = tvec.squeeze()

    return T_camera_board  # i.e. ^camera T_board


def matrix_to_R_t(mat):
    """
    Convert a 4x4 homogeneous matrix to (R, t).
    """
    R = mat[:3, :3]
    t = mat[:3, 3]
    return R, t


def handeye_calibration(R_gripper2base_all, t_gripper2base_all,
                        R_target2cam_all, t_target2cam_all,
                        method=cv2.CALIB_HAND_EYE_TSAI):
    """
    Wrapper for cv2.calibrateHandEye
    Returns R_cam2gripper, t_cam2gripper.
    """
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base_all, t_gripper2base_all,
        R_target2cam_all, t_target2cam_all,
        method=method
    )
    return R_cam2gripper, t_cam2gripper
