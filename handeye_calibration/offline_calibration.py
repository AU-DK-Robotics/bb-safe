import os
import csv
import yaml
import numpy as np
import cv2

class FlowStyleDumper(yaml.SafeDumper):
    """ Custom YAML dumper to format matrices using flow-style lists (bracketed format) """
    def increase_indent(self, flow=False, indentless=False):
        return super(FlowStyleDumper, self).increase_indent(flow=True, indentless=False)

def represent_list_flow(dumper, data):
    """ Force flow-style representation for lists """
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

FlowStyleDumper.add_representer(list, represent_list_flow)


# -------------
# UTILITY FUNCS
# -------------

def load_camera_intrinsics(yaml_path):
    """
    Reads camera intrinsics from a YAML file with fields:
      camera_matrix: {rows: 3, cols: 3, data: [...]}
      dist_coeffs:   {rows: 1, cols: 5, data: [...]}
    Returns: camera_matrix (3x3 np.array), dist_coeffs (1xN np.array)
    """
    with open(yaml_path, 'r') as f:
        doc = yaml.safe_load(f)

    cm_data = doc['camera_matrix']
    dc_data = doc['dist_coeffs']

    # reshape the data into proper matrices
    cm = np.array(cm_data['data'], dtype=np.float64).reshape(cm_data['rows'], cm_data['cols'])
    dc = np.array(dc_data['data'], dtype=np.float64).reshape(dc_data['rows'], dc_data['cols'])
    return cm, dc

def matrix_to_R_t(mat):
    """
    mat is 4x4 or 3x4,
    returns R (3x3) and t(3,)
    """
    R = mat[:3,:3]
    t = mat[:3, 3]
    return R, t

def find_chessboard_corners(image_bgr, pattern_size=(12, 9), draw=False):
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

def create_chessboard_3d_points(pattern_size, square_size):
    """
    Create Nx3 array of the 3D corner coordinates in the board's local frame.
    pattern_size = (cols, rows) # of internal corners
    square_size (float) in meters
    """
    cols, rows = pattern_size
    objp = []
    for r in range(rows):
        for c in range(cols):
            x = c * square_size
            y = r * square_size
            z = 0.0
            objp.append([x, y, z])
    return np.array(objp, dtype=np.float32)

def solve_pnp_for_chessboard(corners, obj_points, camera_matrix, dist_coeffs):
    """
    corners: Nx1x2
    obj_points: Nx3 (float32)
    returns a 4x4 ^camera T_board if successful, else None
    """
    # if corners is None or len(corners) != len(obj_points):
    #     return None
    #
    # success, rvec, tvec = cv2.solvePnP(
    #     obj_points,
    #     corners,
    #     camera_matrix,
    #     dist_coeffs,
    #     flags=cv2.SOLVEPNP_ITERATIVE
    # )
    # if not success:
    #     return None
    #
    # R, _ = cv2.Rodrigues(rvec)
    # T = np.eye(4)
    # T[:3, :3] = R
    # T[:3, 3]  = tvec.squeeze()
    # return T
    # Ensure input shape compatibility
    if corners.ndim == 2 and corners.shape[1] == 2:
        corners_2d_reshaped = corners_2d.reshape(-1, 1, 2)
    else:
        corners_2d_reshaped = corners

    # Use a more robust solvePnP method (IPPE_SQUARE for chessboards)
    # success, rvec, tvec = cv2.solvePnP(obj_points, corners_2d_reshaped,
    #                                    camera_matrix, dist_coeffs,
    #                                    flags=cv2.SOLVEPNP_ITERATIVE)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(obj_points, corners_2d_reshaped,
                                       camera_matrix, dist_coeffs,reprojectionError=8.0,
                                       flags=cv2.SOLVEPNP_ITERATIVE)

    if not success or not inliers.any():
        return None  # SolvePnP failed

    print(len(inliers))

    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # # Normalize R to ensure it's a valid rotation matrix
    # U, _, Vt = np.linalg.svd(R)
    # R = U @ Vt  # This ensures R is a proper rotation matrix

    # Build 4x4 homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.squeeze()

    return T

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

# -------------
# MAIN OFFLINE CALIBRATION
# -------------
def main():
    # 1) Load camera intrinsics from YAML
    intr_yaml_path = "captures_1280_720/final_calibration.yaml"  # adjust
    camera_matrix, dist_coeffs = load_camera_intrinsics(intr_yaml_path)
    print("Loaded camera matrix:\n", camera_matrix)
    print("Dist coeffs:", dist_coeffs)

    # 2) Configuration for the checkerboard
    # For example, if you have a 13x10 internal corners with 15mm squares:
    pattern_size = (12, 9)
    square_size = 0.015  # meters
    object_points_3d = create_chessboard_3d_points(pattern_size, square_size)

    # 3) Read the CSV with robot poses
    #    Suppose each row is: capture_idx, base_T_ee(0,0), base_T_ee(0,1), ... base_T_ee(2,3)
    csv_path = "captures_1280_720/poses.csv"
    image_folder = "captures_1280_720"  # folder containing capture_1.png, capture_2.png, etc.

    # We'll store data for hand-eye calibration
    R_gripper2base_all = []
    t_gripper2base_all = []
    R_target2cam_all   = []
    t_target2cam_all   = []

    with open(csv_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader, None)  # optional if there's a header row
        for row_idx, row in enumerate(reader):
            # row example: [capture_index, base_T_ee(0,0), ..., base_T_ee(2,3)]
            capture_idx = int(row[0])
            # the next 12 values for a 3x4, or 16 values if 4x4 was stored
            # let's assume 3x4 flatten for simplicity
            mat_vals = list(map(float, row[1:]))

            if len(mat_vals) == 12:
                # Reconstruct 3x4, then 4x4
                base_T_ee_3x4 = np.array(mat_vals, dtype=np.float64).reshape(3,4)
                base_T_ee = np.vstack([base_T_ee_3x4, [0,0,0,1]])
            elif len(mat_vals) == 16:
                # Already a 4x4
                base_T_ee = np.array(mat_vals, dtype=np.float64).reshape(4,4)
            else:
                print(f"Invalid matrix size in CSV row #{row_idx+1}. Skipping.")
                continue

            # We'll invert base_T_ee to get ee->base (gripper->base)
            ee_T_base = np.linalg.inv(base_T_ee)

            # 4) Load the corresponding image
            #    e.g. "capture_1.png" if capture_idx=1
            img_name = f"capture_{capture_idx}.png"
            img_path = os.path.join(image_folder, img_name)
            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}. Skipping.")
                continue

            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"Failed to load image: {img_path}. Skipping.")
                continue

            # gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # 5) Find chessboard corners
            found, corners = find_chessboard_corners(img_bgr, pattern_size)
            if not found:
                print(f"No corners found in {img_name}. Skipping.")
                continue

            # 6) Estimate ^camera T_board with solvePnP
            camera_T_board = solve_pnp_for_chessboard(corners, object_points_3d,
                                                      camera_matrix, dist_coeffs)
            if camera_T_board is None:
                print(f"solvePnP failed for {img_name}. Skipping.")
                continue

            # Invert for board->camera (target->cam)
            board_T_camera = np.linalg.inv(camera_T_board)

            # 7) Convert transforms to R,t for calibrateHandEye
            R_ee2base, t_ee2base = matrix_to_R_t(base_T_ee)
            R_board2cam, t_board2cam = matrix_to_R_t(camera_T_board)

            R_gripper2base_all.append(R_ee2base)
            t_gripper2base_all.append(t_ee2base)
            R_target2cam_all.append(R_board2cam)
            t_target2cam_all.append(t_board2cam)

    # 8) Run hand-eye calibration
    if len(R_gripper2base_all) < 2:
        print("Not enough valid data for hand-eye calibration.")
        return

    print("\nRunning cv2.calibrateHandEye ...")
    # You can pick any method: cv2.CALIB_HAND_EYE_TSAI, etc.
    R_cam2ee, t_cam2ee = cv2.calibrateHandEye(
        R_gripper2base_all, t_gripper2base_all,
        R_target2cam_all,   t_target2cam_all,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    # Build the 4x4 transform
    ee2cam = np.eye(4)
    ee2cam[:3,:3] = R_cam2ee
    ee2cam[:3, 3] = t_cam2ee.ravel()

    cam2ee = np.linalg.inv(ee2cam)

    print("==== Results ====")
    print("Camera->End-Effector (cam2ee):\n", cam2ee)
    print("End-Effector->Camera (ee2cam):\n", ee2cam)

    # 9) Optionally, save the result to a new YAML
    out_yaml = "handeye_offline.yaml"
    save_calibration_yaml(
        yaml_path=out_yaml,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        cam2ee_4x4=cam2ee,
        ee2cam_4x4=ee2cam
    )
    print(f"\nSaved extrinsic transform to {out_yaml}.")


if __name__ == "__main__":
    main()
