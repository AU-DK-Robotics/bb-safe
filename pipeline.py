#!/usr/bin/env python3

from admittance_control.admittance_controller import ComputeAdmittance
import time
from time import sleep
import numpy as np
import math
import platform
from collections import deque
from enum import StrEnum,Enum
from typing import Dict, List, Optional
from PIL import Image
import sys
from datetime import datetime
import csv
from pathlib import Path
from uuid import uuid4 as uuid
from contextlib import nullcontext

# UR_RTDE
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

# Gripper
import serial
from serial import Serial
from serial.tools.list_ports import comports

# Camera
from handeye_calibration.camera_interface_async import RealSenseInterfaceAsync as RealSenseInterface

# Object dection
from detect_yolo import detectorYOLO
from object_detection.coordinate_transformation import transform_to_robot_frame
import cv2

def variableAdmittanceMoveL(rtde_c, rtde_r, pose_end, T, dt, M, C, K,
                            desired_z_force = 0.0, out_dir = None,
                            vac_distance_threshold=0, K_fac=np.ones(6), C_fac=np.ones(6),
                            force_lowpass_alpha=0.2, pose_start = np.array([]),
                            pos_err_threshold=0.001, zero_ft = True):

    csv_fields = ["Time",
                  "Iteration",
                  "Ratio",
                  "Pose x", "Pose y", "Pose z", "Pose Rx", "Pose Ry", "Pose Rz",
                  "Ref. pose x", "Ref. pose y", "Ref. pose z", "Ref. pose Rx", "Ref. pose Ry", "Ref. pose Rz",
                  "Trg. pose x", "Trg. pose y", "Trg. pose z", "Trg. pose Rx", "Trg. pose Ry", "Trg. pose Rz",
                  "Offset x", "Offset y", "Offset z", "Offset Rx", "Offset Ry", "Offset Rz",
                  "Vx", "Vy", "Vz", "wx", "wy", "wz",
                  "Fx", "Fy", "Fz", "Mx", "My", "Mz",
                  "K11", "K22", "K33", "K44", "K55", "K66",
                  "C11", "C22", "C33", "C44", "C55", "C66"]

    if not pose_start.size:
        pose_start = np.array(rtde_r.getActualTCPPose())

    if out_dir:
        t_safe = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        csv_file = out_dir / ("admittance_log_"+t_safe+".csv")
        adm_settings = out_dir / ("admittance_settings_"+t_safe+".txt")

        with adm_settings.open("w") as f:
            f.write(f"# T: {T}\n# dt: {dt}\n# pose_start: {pose_start.tolist()}\n# pose_end: {pose_start.tolist()}\n# M:{M.tolist()}\n# VAC distance thres.: {vac_distance_threshold}\n# K_fac: {K_fac}\n# C_fac: {C_fac}\n# Stopping threshold: {pos_err_threshold}\n# Desired z-force: {desired_z_force}\n")

    with csv_file.open("w") if out_dir else nullcontext() as f:
        if f:
            csv_writer = csv.DictWriter(f,csv_fields,delimiter=";")
            csv_writer.writeheader()

        controller = ComputeAdmittance(M, C, K, dt)
        state = np.zeros(12)
        ratio = 0

        K_upd = K.copy()
        C_upd = C.copy()

        if zero_ft:
            rtde_c.zeroFtSensor()

        force = np.zeros(6)
        force_mag = np.zeros(3)
        desired_force = np.zeros(6)
        desired_force[2] = desired_z_force
        tau_ext_mag = 0.0
        force_z_err = 0.0
        filtered_force = np.zeros(6)

        t0 = time.perf_counter_ns()
        t_prev = t0
        itercount = 0

        # ignore rotations
        pose_end[3:6] = pose_start[3:6]

        targ_error = np.zeros(6)

        while not (ratio > 0.9 and np.abs(z_error) < 0.0015 and np.abs(force_z_err) < 1.0):

            # Timing
            t_start = rtde_c.initPeriod()
            t_now = time.perf_counter_ns()
            delta_t = (t_now - t_prev)/1e9

            # Position reading
            curr_pose = np.array(rtde_r.getActualTCPPose())
            z_error = curr_pose[2] - pose_end[2]

            # Force reading
            force = rtde_r.getActualTCPForce()
            if force_lowpass_alpha:
                filtered_force = lowPassFilter(np.array(force),filtered_force,force_lowpass_alpha)
            else:
                filtered_force = force
            force_z_err = filtered_force[2] - desired_z_force

            t_prev = t_now
            t = (t_now - t0)/1e9
            if t < T:
                ratio = t / T
            else:
                ratio = 1.0
            itercount += 1

            tau_ext = filtered_force - desired_force

            reference_pose = pose_start + ratio * (pose_end - pose_start)

            # Variable admittance
            curr_pose = np.array(rtde_r.getActualTCPPose())
            pos_error = np.linalg.norm(curr_pose[:3] - pose_end[:3])
            pos_error_ref = np.linalg.norm(reference_pose[:3] - pose_end[:3])


            if vac_distance_threshold:
                if z_error < vac_distance_threshold:
                    if np.any(K_fac - 1):
                        K_scale = 1 + (K_fac - 1) * (1 - z_error / vac_distance_threshold)
                        for i in range(6): K_upd[i, i] = K[i, i] * K_scale[i]
                    if np.any(C_fac - 1):
                        C_scale = 1 + (C_fac - 1) * (1 - z_error / vac_distance_threshold)
                        for i in range(6): C_upd[i, i] = C[i, i] * C_scale[i]
                    controller.update_matrices(M, C_upd, K_upd)




            state = controller(tau_ext, state)
            offset = state[:6]       # position offset computed by the controller
            offset_dist = np.linalg.norm(offset[0:3])
            velocity = state[6:12]   # computed velocity (the second half)
            target_pose = reference_pose + offset

            val = [t, itercount, ratio]
            val.extend(curr_pose.tolist())
            val.extend(reference_pose.tolist())
            val.extend(target_pose.tolist())
            val.extend(offset.tolist())
            val.extend(velocity.tolist())
            val.extend(filtered_force)
            val.extend(np.diag(K_upd).tolist())
            val.extend(np.diag(C_upd).tolist())
            data = dict(zip(csv_fields2,val))

            if f:
                csv_writer.writerow(data)

            rtde_c.servoL(target_pose.tolist(), 0.1, 0.5, dt, 0.03, 300)
            rtde_c.waitPeriod(t_start)
        rtde_c.servoStop()

def getPoseMatrix(rtde_r):
    """
    Return ^base T_ee as a 4x4 homogeneous transformation matrix.
    UR 'getActualTCPPose()' returns [x, y, z, Rx, Ry, Rz] in meters / axis-angle.
    """
    tcp_pose = rtde_r.getActualTCPPose()
    x, y, z, rx, ry, rz = tcp_pose

    # Convert axis-angle (rotation vector) to rotation matrix
    rotation_vector = np.array([rx, ry, rz], dtype=float)
    R, _ = cv2.Rodrigues(rotation_vector)  # Convert to 3x3 rotation matrix

    # Build the 4x4 homogeneous transformation matrix
    pose_mat = np.eye(4)
    pose_mat[:3, :3] = R
    pose_mat[:3, 3] = [x, y, z]

    return pose_mat

def lowPassFilter(new_value, prev_filtered, alpha):
    """
    Exponential arm_moving average (EMA) low-pass filter for vector data.

    Parameters:
      new_value (np.ndarray): The latest 6D measurement.
      prev_filtered (np.ndarray): The previous filtered 6D value.
      alpha (float): Filter coefficient in [0, 1]. Lower values yield more smoothing.

    Returns:
      np.ndarray: The updated filtered value.
    """
    return alpha * new_value + (1 - alpha) * prev_filtered

def urMoveJ(rtde_c,data,speed=0.25,post_move_wait=1,isIK=False):
    if isIK:
        rtde_c.moveJ_IK(data,speed=speed,asynchronous=True)
    else:
        rtde_c.moveJ(data,speed=speed,asynchronous=True)
    while rtde_c.getAsyncOperationProgress()>=0:
        continue
    time.sleep(post_move_wait)
    return data

def gripperSerialCmd(ard,msg,enc="ASCII",verbose=False):
    if verbose:
        print("Wrote: {}".format(msg))
    ard.write(bytes(msg,encoding=enc))
    res = ard.readline().decode(encoding='ASCII')
    if verbose:
        if res:
            # Response already includes a newline
            print("Read: {}".format(res),end="")
        else:
            print("No response")
    return res

def getGripperSensors(ard,rtde_r):
    res = gripperSerialCmd(ard,"5")
    if res:
        val = res.split(", ")
        t = int(val[0])
        sens = val[3].split(" ")
        dist = int(sens[0])

        # Apply invisible boundary representing the top of the breeding blanket
        T_base_tcp = getPoseMatrix(rtde_r)
        tcp_z = T_base_tcp[2,3]
        if tcp_z >= 0.35:
            T_base_interface = np.eye(4)
            T_base_interface[0,3] = 0.32   # x
            T_base_interface[1,3] = -0.21  # y
            T_base_interface[2,3] = 0.1365 # z
            T_interface_base = np.linalg.inv(T_base_interface)
            T_tcp_ultra = np.eye(4)
            T_tcp_ultra[1,3] = 0.077 # y
            T_tcp_ultra[2,3] = 0.030 # z
            T_base_ultra = np.matmul(T_base_tcp,T_tcp_ultra)
            T_interface_ultra = np.matmul(T_interface_base,T_base_ultra)

            # interface dimensions
            l = 0.197   # x
            s = 0.13702 # y
            c = 0.036   # corner chamfer

            # ultrasound coords relative to interface
            x = T_interface_ultra[0,3]
            y = T_interface_ultra[1,3]
            z = T_interface_ultra[2,3]

            # print(T_base_ultra)

            bounds = [ np.abs(y) > s/2,
                    np.abs(x) > l/2,
                    y > -x + l/2 - c + s/2,
                    y >  x + l/2 - c + s/2,
                    y < -x - l/2 + c - s/2,
                    y <  x - l/2 + c - s/2 ]

            # Assume vertical gripper orientation
            if np.any(bounds):
                print(f"Restricting distance measurement due to out of bounds: {[int(i) for i in bounds]}")
                dist = round(z*100)

        arms = (float(sens[1]), float(sens[2])) # R, L
        return t, dist, arms
    else:
        return None, None, None

def connectGripper(serial_device, sertimeout):
    # If no device given, pick one automatically
    if not serial_device:
        serdev = [i.device for i in comports()]
        if platform.system() == "Linux":
            serdev = [i for i in serdev if i.startswith("/dev/ttyACM")]
    if serdev:
        serial_device = serdev[0]
    else:
        raise Exception("Could not find any Arduino serial devices")

    print("Connecting to gripper (serial device " + serial_device + ")... ",end="")

    ser = serial.Serial(serial_device,timeout=sertimeout,baudrate=115200,write_timeout=sertimeout)

    # Make sure the Arduino is responding consistently
    for _ in range(5):
        res = gripperSerialCmd(ser,"0",verbose=False)
    if not res:
        raise(Exception("Arduino not responding"))
    print("OK")

    return ser

def initializeConnections(robot_ip, freq, hec_path, out_dir, serial_device = None, sertimeout = 1):

    # Connect to gripper (Arduino)
    ser = connectGripper(serial_device, sertimeout)

    # Connect to UR robot
    print("Connecting to UR RTDE receive and control interfaces... ",end="")
    rtde_c = RTDEControlInterface(robot_ip, freq)
    rtde_r = RTDEReceiveInterface(robot_ip, freq)
    print("OK")

    # Connect to camera
    print("Connecting to RealSense camera... ",end="")
    camera = RealSenseInterface(hec_path,out_dir)
    print("OK")

    return ser, rtde_c, rtde_r, camera

def getRandomViewQ(rtde_c, viewP, viewQ, spread, log_path=""):
    while True:
        randPose, xy = getRandomPoseXY(viewP, spread)
        print(xy)
        if rtde_c.getInverseKinematicsHasSolution(randPose): break
    randQ = rtde_c.getInverseKinematics(randPose, qnear=viewQ)

    msg = f"View pose: {randPose.tolist()} (X-Y offset: {xy.tolist()})"
    if log_path:
        with log_path.open("a") as f:
            f.write(msg + "\n")
    print(msg)
    urMoveJ(rtde_c, rtde_c.getInverseKinematics(randPose,qnear=viewQ))
    return randQ

def getRandomPoseXY(pose, spread):
    rand_gen = np.random.default_rng()
    xy = 2*rand_gen.random(2)-1
    xy = xy * spread
    pose_delta = np.concatenate([xy,np.zeros(4)])
    randPose = pose + pose_delta
    return randPose, xy

def detectInterface(camera, detector, rtde_r, spread=0.0, interface_type="big_interface", attempts=3,
                    detection_save_path = "", depth_save_path = "", img_save_path="", log_path=""):

    print("Detecting interface")

    if not img_save_path:
        img_i_save_path = ""
    for i in range(attempts):
        print(f"Attempt {i}")
        color_image, depth_image, depth_colormap = camera.get_frames()

        if detection_save_path:
            det_i_save_path = detection_save_path.with_stem(detection_save_path.stem + f"_{i}")
        if img_save_path:
            img_i_save_path = img_save_path.with_stem(img_save_path.stem + f"_{i}")
        if depth_save_path:
            np.save(depth_save_path.with_stem(depth_save_path.stem + f"_{i}").with_suffix(".npy"),depth_image,allow_pickle=False)

            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=255/depth_image.max()),
                cv2.COLORMAP_JET
            )
            cv2.imwrite(depth_save_path, depth_colormap)

        _,detections = detector.detect_objects(color_image, depth_image, img_save_path=img_i_save_path, log_path=log_path)

        matches = list(i for i in detections if i["label"] == interface_type)
        if len(matches) == 1:
            match = True
        else:
            match = False

        display_image = np.array([])
        # display_image = depth_colormap
        if detection_save_path:
            save_image = color_image.copy()

        for obj in detections:

            x, y, w, h = (round(i) for i in obj["bbox"])
            cx, cy = (round(i) for i in obj["center"])

            baseTee_matrix = getPoseMatrix(rtde_r)

            depth = obj["depth"]

            obj_coords_base, obj_coords_cam, obj_coords_ee = transform_to_robot_frame(obj["center"], depth, baseTee_matrix,camera.camera_matrix,camera.T_cam_matrix)
            obj_x, obj_y, obj_z = obj_coords_base  # Extract X, Y, Z

            if display_image.size:
                cv2.rectangle(display_image, (x, y), (x + w, y + h), [0,255,0], 2)
                cv2.putText(display_image, f"Class: {obj["class_id"]} | ({obj_x:.3f}, {obj_y:.3f}, {obj_z:.3f}) | Depth: {depth:.3f}",
                            (x, y + h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,255,0], 2)


            if detection_save_path:
                cv2.rectangle(save_image, (x, y), (x + w, y + h), [0,255,0], 2)
                cv2.putText(save_image, f"Class: {obj["class_id"]} | ({obj_x:.3f}, {obj_y:.3f}, {obj_z:.3f})",
                            (x, y + h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,255,0], 2)

            if match and obj["label"] == interface_type:
                if display_image.size:
                    cv2.circle(display_image,obj["depth_spot"], 3, [0,255,0],-1) # negative thickness = filled
                align_pose = np.array([obj_x, obj_y, 0.36, 0, math.pi, 0])

        if display_image.size:
            Image.fromarray(display_image[:, :, ::-1]).show()

        if detection_save_path:
            cv2.imwrite(det_i_save_path, save_image)


        # cv2.imshow("RealSense VLA Detection", display_image)
        # cv2.waitKey(1)

        if match:
            if spread:
                align_pose, xy_offset = getRandomPoseXY(align_pose, spread)
            else:
                xy_offset = np.zeros(2)
            insert_pose = align_pose.copy()
            insert_pose[2] = 0.216
            msg = f"Alignment pose: {align_pose.tolist()} (X-Y offset: {xy_offset.tolist()})\nInsertion pose: {insert_pose.tolist()}"
            print(msg)
            if log_path:
                with log_path.open("a") as f:
                    f.write(msg+"\n")
            return align_pose, insert_pose

    raise Exception(f"No detections after {attempts} attempts")

def engageGripper(ard,cmd,movetime,post_move_wait=1):
    if cmd:
        gripperSerialCmd(ard,"2")
    else:
        gripperSerialCmd(ard,"1")
    time.sleep(movetime)
    time.sleep(post_move_wait)

class EvalPrefix(StrEnum):
    SCENE  = "based on the image, evaluate the whole interface is visible or not, then plan the robotic actions for gripper engagement of transporter."
    ALIGNMENT  = "based on the image, and the distance to interface {ultrasonic_dis_cm}, evaluate if the alignment between the gripper and interface is good or not. then plan the following actions for engagement."
    INSERTION  = "based on the image, contact wrench {tcp_wrench_N_Nm}, and the distance to interface {ultrasonic_dis_cm}, evaluate if the insertion between the gripper and interface is good or not. then plan the following actions for engagement."
    ENGAGEMENT = "based on the image, the distance to interface {ultrasonic_dis_cm}, and the folding arm force {force_gauge}, evaluate the engagement between the folding arm and interface is good or not. then plan the following actions for engagement."

def stringify_wrench(w: List[float]) -> str:
    if not isinstance(w, (list, tuple)) or len(w) < 6:
        return "unknown"
    fx, fy, fz, tx, ty, tz = w[:6]
    return f"[Fx={fx:.2f} N, Fy={fy:.2f} N, Fz={fz:.2f} N, Tx={tx:.3f} Nm, Ty={ty:.3f} Nm, Tz={tz:.3f} Nm]"

def stringify_force_gauge(v) -> str:
    if v is None:
        return "unknown"
    if isinstance(v, (list, tuple)):
        try:
            return f"[left={float(v[0]):.2f}, right={float(v[1]):.2f}]"
        except Exception:
            return str(v)
    try:
        return f"{float(v):.2f}"
    except Exception:
        return str(v)

def stringify_distance(d) -> str:
    if d is None:
        return "unknown"
    try:
        return f"{float(d):.2f} cm"
    except Exception:
        return str(d)

def evaluateScene(model, camera, eval_mode, img_save_path="", log_path=""):
    succ_actions = " planned actions: detect interface; align gripper with interface; evaluate alignment; insert gripper; evaluate insertion; engage gripper; evaluate engagement."
    fail_actions = " planned actions: chase interface; evaluate scene."

    succ = True
    if eval_mode[0] == 0:
        msg = "No evaluation."
    elif eval_mode[0] == 1:
        msg = f"Heuristic evaluation: no heuristics, success: {succ}."
    response = msg + succ_actions
    msg = response

    print(msg)
    if log_path:
        with log_path.open("a") as f:
            f.write(msg+"\n")
    return response, uuid()

def evaluateAlignment(model, camera, ard, rtde_r, eval_mode, img_save_path="", log_path=""):
    t, dist, arms = getGripperSensors(ard, rtde_r)
    dist_str = stringify_distance(dist)

    succ_actions = " planned actions: insert gripper; evaluate insertion; engage gripper; evaluate engagement."
    fail_actions = " planned actions: chase interface; detect interface; align gripper with interface; evaluate alignment."

    response, uid = ["", ""]
    if eval_mode[1] == 0:
        succ = True
        msg = "No evaluation."
    elif eval_mode[1] == 1:
        succ = dist <= (22-3)
        msg = f"Heuristic evaluation: distance for aligment: {dist_str}, success: {succ}."
    if succ:
        response = msg + succ_actions
    else:
        response = msg + fail_actions
    msg = response
    print(msg)
    if log_path:
        with log_path.open("a") as f:
            f.write(msg+"\n")
    return response, uuid()

def evaluateInsertion(model, camera, rtde_r, ard, eval_mode, img_save_path="", log_path=""):
    force = rtde_r.getActualTCPForce()
    t, dist, arms = getGripperSensors(ard, rtde_r)
    force_str = stringify_wrench(force)
    dist_str = stringify_distance(dist)

    succ_actions = " planned actions: engage gripper; evaluate engagement."
    fail_actions = " planned actions: remove gripper; insert gripper; evaluate insertion."

    response, uid = ["", ""]
    if eval_mode[2] == 0:
        succ = True
        msg = "No evaluation."
    elif eval_mode[2] == 1:
        eval1 = (dist > 3) and (dist < 5)
        eval2 = force[2] >= 4.0
        succ = eval1 and eval2
        msg = f"Heuristic evaluation: distance for insertion: {dist_str}, F-T for insertion: {force_str}, success: {succ}."
    if succ:
        response = msg + succ_actions
    else:
        response = msg + fail_actions
    msg = response
    print(msg)
    if log_path:
        with log_path.open("a") as f:
                f.write(msg+"\n")
    return response, uuid()

def evaluateEngagement(model, camera, ard, rtde_r, eval_mode, img_save_path="", log_path=""):
    t, dist, arms = getGripperSensors(ard, rtde_r)
    arms_str = stringify_force_gauge(arms)
    dist_str = stringify_distance(dist)

    succ_actions = " planned actions: finished."
    fail_actions = " planned actions: disengage gripper; engage gripper; evaluate engagement."

    response, uid = ["", ""]
    if eval_mode[3] == 0:
        succ = True
        msg = "No evaluation."
    if eval_mode[3] == 1:
        eval1 = np.any(np.array(arms) >= 4.0)
        eval2 = (dist > 3) and (dist < 5)
        succ = eval1 and eval2
        msg = f"Heuristic evaluation: arm forces for engagement: {arms_str}, distance for engagement: {dist_str}, success: {succ}."
    if succ:
        response = msg + succ_actions
    else:
        response = msg + fail_actions
    msg = response
    print(msg)
    if log_path:
        with log_path.open("a") as f:
            f.write(msg+"\n")
    return response, uuid()


def evaluate(model, camera, prefix, img_save_path = "", log_path = ""):
    img_rgb, _, _ = camera.get_frames()
    model.setMode("vqa")
    response, _ = model.infer(img_rgb, prefix, img_save_path = img_save_path, log_path = log_path)
    return response

def finalEvaluation(rtde_r, ard, eval_mode, now, spread, csv_path = Path("")):
    force = rtde_r.getActualTCPForce()
    t, dist, arms = getGripperSensors(ard, rtde_r)
    force_str = stringify_wrench(force)
    dist_str = stringify_distance(dist)
    arms_str = stringify_force_gauge(arms)

    eval1 = np.any(np.array(arms) >= 4.0)
    eval2 = (dist > 3) and (dist < 5)

    succ = eval1 and eval2
    msg = f"Final heuristic evaluation: distance: {dist_str}, arm forces: {arms_str}.\nFinal success: {succ}."
    print(msg)
    if csv_path.name:
        data = [now, "YOLO"]
        data.extend(eval_mode)
        data.append(spread)
        data.append(dist)
        data.extend(arms)
        data.extend(force)
        data.append(int(succ))
        with csv_path.open("a") as f:
            writer = csv.writer(f, delimiter = ";")
            writer.writerow(data)

def main(ldict,action_adapter):

    prev_response_id = ""

    log_dir = ldict["out_dir"]
    ldict["log"] = log_dir / "log.txt"
    log_dir.mkdir(parents=True)
    rgb_dir = log_dir / "img_rgb"
    rgb_dir.mkdir()
    z_dir = log_dir / "img_z"
    z_dir.mkdir()
    detect_dir = log_dir / "detections"
    detect_dir.mkdir()

    print(f"Output directory: {ldict["out_dir"]}")

    while True:
        now = datetime.now()
        t = now.strftime("%Y-%m-%d %H:%M:%S")
        t_safe = now.strftime("%Y-%m-%d_%H-%M-%S")
        ldict["rgb_save_path"] = rgb_dir / (t_safe + ".png")
        ldict["z_save_path"] = z_dir / (t_safe + ".png")
        ldict["detect_save_path"] = detect_dir / (t_safe + ".png")

        if prev_response_id != ldict["response_id"]:
            prev_response_id = ldict["response_id"]
            status_actionplan = ldict["response"].strip()[:-1].split(" planned actions: ")
            if status_actionplan != ldict["response"]: # successful split
                status = status_actionplan[0]
                actionplan = status_actionplan[1].split("; ")
                if actionplan != status_actionplan[1]:
                    actionplan = deque(actionplan)
                    continue
            raise Exception("Failed to parse response")

        if actionplan:
            a = actionplan.popleft() # A string
            msg = f"Timestamp: {t}\nAction: {a}\n"
            print(msg,end="")
            with ldict["log"].open("a") as log:
                log.write(msg)
            exec(action_adapter[a],globals(),ldict)
            continue

        print("No more actions")
        break

    print(f"Output directory: {ldict["out_dir"]}")

if __name__ == '__main__':

    # Adapt action descriptions to code snippets
    action_adapter = {
        "move to view pose":            "rand_view_q = getRandomViewQ(rtde_c, viewP, viewQ, spread = rand_spread_scene)",
        "evaluate scene":               "response, response_id = evaluateScene(model, camera, eval_mode, img_save_path=rgb_save_path, log_path=log)",
        "chase interface":              "rand_view_q = urMoveJ(rtde_c, getRandomViewQ(rtde_c, viewP, viewQ, spread = rand_spread_scene))",
        "detect interface":             "align_pose, insert_pose = detectInterface(camera, model, rtde_r, spread = rand_spread_align, detection_save_path = detect_save_path, depth_save_path = z_save_path, img_save_path=rgb_save_path, log_path=log)",
        "align gripper with interface": "urMoveJ(rtde_c, align_pose, isIK=True)",
        "evaluate alignment":           "response, response_id = evaluateAlignment(model, camera, ser, rtde_r, eval_mode, img_save_path=rgb_save_path, log_path=log)",
        "insert gripper":               "variableAdmittanceMoveL(rtde_c, rtde_r, insert_pose, 20.0, dt, admit_M, insert_C, insert_K, K_fac = insert_K_fac, C_fac = insert_C_fac, desired_z_force = insert_Fz, vac_distance_threshold = 0.01",
        "evaluate insertion":           "response, response_id = evaluateInsertion(model, camera, rtde_r, ser, eval_mode, img_save_path=rgb_save_path, log_path=log)",
        "remove gripper":               "variableAdmittanceMoveL(rtde_c, rtde_r, align_pose, 10.0, dt, admit_M, remove_C, remove_K, zero_ft = False, out_dir=out_dir))",
        "engage gripper":               "engageGripper(ser, True, servo_time)",
        "evaluate engagement":          "response, response_id = evaluateEngagement(model, camera, ser, rtde_r, eval_mode, img_save_path=rgb_save_path, log_path=log)",
        "disengage gripper":            "engageGripper(ser, False, servo_time)",
        "finished":                     """finalEvaluation(rtde_r, ser, eval_mode, now, (rand_spread_scene, rand_spread_align), csv_path = global_csv)
engageGripper(ser, False, servo_time)
"variableAdmittanceMoveL(rtde_c, rtde_r, align_pose, 10.0, dt, admit_M, remove_C, remove_K, zero_ft = False , out_dir=out_dir))"""
    }

    # Initialize dictionary of local variables for exec() and insert constants
    ldict = {}

    # Gripper actuation time
    servo_spr = 8    # time per rev at the configured speed (sec)
    rack_stroke = 22 # Stroke distance for engaging gripper (mm)
    spur_diam = 12   # Pitch diameter for spur gear (mm)
    rev_per_stroke = rack_stroke/(math.pi*spur_diam)
    ldict["servo_time"] = servo_spr*rev_per_stroke

    # Experiment time
    ldict["now"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Hand-eye calibration file location
    ldict["hec_path"] = "handeye_calibration/captures_1280_720_20250908_1/final_calibration.yaml"

    # IP address of robot
    ldict["robot_ip"] = "192.168.1.254"

    # Timing, for RTDE and admittance
    ldict["freq"] = 500
    ldict["dt"] = 1 / ldict["freq"]
    ldict["dt_ns"] = ldict["dt"]*1e9

    # Maximum x- and y-axis random offset magnitudes for scene evaluation / object detection and alignment poses
    ldict["rand_spread_scene"] = 0.04
    ldict["rand_spread_align"] = 0.04

    # Type of evaluation to do after each action (detection, alignment, insertion, engagement)
    # 0 = none, 1 = rule-based
    ldict["eval_mode"] = [1, 1, 1, 1]

    # Admittance parameters
    ldict["admit_M"] = np.diag([50, 50, 50, 50, 50, 50])
    ldict["insert_C"] = np.diag([250, 250, 1000, 1000, 1000, 1000])
    ldict["insert_K"] = np.diag([0, 0, 400, 1000, 1000, 1000])
    ldict["insert_C_fac"] = np.array([8, 8, 10, 1, 1, 1])
    ldict["insert_K_fac"] = np.array([1, 1, 5, 1, 1, 1])
    ldict["remove_C"] = np.diag([2000, 2000, 1000, 10000, 10000, 10000])
    ldict["remove_K"] = np.diag([500, 500, 500, 1000, 1000, 1000])

    # Desired gripper insertion force (TCP z-axis)
    ldict["insert_Fz"] = 8.0

    # Root folder for storing experiment data
    global_dir = Path.home() / "bb_safe" / "Experiments"
    global_dir.mkdir(exist_ok=True, parents=True)

    # CSV file with overview of successful experiments
    ldict["global_csv"] = global_dir / "global.csv"

    # Where to store data from this experiment
    eval_mode_str = ''.join([str(i) for i in ldict["eval_mode"]])
    ldict["out_dir"] = global_dir / ("YOLO_" + eval_mode_str) / ldict["now"]

    # Starting action sequence and its UUID
    ldict["response_id"] = "6790e196-fff0-419b-ad6a-32e4b5955a02"
    ldict["response"] = "starting. planned actions: chase interface; evaluate scene."

    # Initialize devices
    ldict["ser"], ldict["rtde_c"], ldict["rtde_r"], ldict["camera"] = initializeConnections(ldict["robot_ip"], ldict["freq"], ldict["hec_path"], ldict["out_dir"])

    # Hard-coded start pose
    ldict["viewQ"] = [1.8504924774169922, -1.4910245326212426, 0.5884845892535608, -0.6688453716090699, -1.5668700377093714, -0.5082209745990198]
    ldict["viewP"] = ldict["rtde_c"].getForwardKinematics(ldict["viewQ"],tcp_offset=np.zeros(6).tolist())

    # Load the object detection model (YOLO)
    ldict["model"] = detectorYOLO(model_weights_path="object_detection/runs/detect/train2/weights/best.pt",confidence_threshold=0.9)

    # Start the main loop while accepting keyboard interrupts
    try:
        main(ldict,action_adapter)
    except KeyboardInterrupt:
        print("Received keyboard interrupt, stopping")
    finally:
        camera.stop()
