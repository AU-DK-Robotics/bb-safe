#!/usr/bin/env python3

from admittance_control.admittance_controller import ComputeAdmittance
import time
from time import sleep
import numpy as np
import math
from collections import deque
from enum import StrEnum
from PIL import Image
from datetime import datetime
import csv
from pathlib import Path
from uuid import uuid4 as uuid
from contextlib import nullcontext

# UR_RTDE
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

# Gripper
from gripper_control import gripper

# Camera
from camera_utils.camera_interface_async import RealSenseInterfaceAsync as RealSenseInterface
from camera_utils import snow

# Object dection
from detect_yolo import detectorYOLO
from object_detection.coordinate_transformation import transform_to_robot_frame
import cv2

# Evaluation model
from evaluate_vlm import VLM

# Sensors
import sensing

def variableAdmittanceMoveL(rtde_c, rtde_r, pose_end, T, dt, M, C, K,
                            desired_z_force = 0.0, out_dir = None,
                            vac_distance_threshold=0, K_fac=np.ones(6), C_fac=np.ones(6),
                            force_lowpass_alpha=0.2, pose_start = np.array([]),
                            pos_err_threshold=0.001, zero_ft = True, std = 0):

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

    t_safe = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_file = Path("admittance_log_"+t_safe+".csv")
    if out_dir:
        csv_file = out_dir / csv_file
        adm_settings = out_dir / ("admittance_settings_"+t_safe+".txt")

        with adm_settings.open("w") as f:
            f.write(f"# T: {T}\n# dt: {dt}\n# pose_start: {pose_start.tolist()}\n# pose_end: {pose_start.tolist()}\n# M:{M.tolist()}\n# VAC distance thres.: {vac_distance_threshold}\n# K_fac: {K_fac}\n# C_fac: {C_fac}\n# Stopping threshold: {pos_err_threshold}\n# Desired z-force: {desired_z_force}\n")

    with csv_file.open("w") if out_dir else nullcontext() as f:
        if f:
            csv_writer = csv.DictWriter(f,csv_fields,delimiter=";")
            csv_writer.writeheader()
        else:
            csv_writer = None

        controller = ComputeAdmittance(M, C, K, dt)
        state = np.zeros(12)
        ratio = 0

        K_upd = K.copy()
        C_upd = C.copy()

        if zero_ft:
            rtde_c.zeroFtSensor()

        desired_force = np.zeros(6)
        desired_force[2] = desired_z_force
        force_z_err = 0.0
        filtered_force = np.zeros(6)

        t0 = time.perf_counter_ns()
        itercount = 0

        # ignore rotations
        pose_end[3:6] = pose_start[3:6]
        curr_pose = np.array(rtde_r.getActualTCPPose())
        z_error = curr_pose[2] - pose_end[2]

        while not (ratio > 0.9 and np.abs(z_error) < 0.0015 and np.abs(force_z_err) < 1.0):

            # Timing
            t_start = rtde_c.initPeriod()
            t_now = time.perf_counter_ns()

            # Position reading
            curr_pose = np.array(rtde_r.getActualTCPPose())
            z_error = curr_pose[2] - pose_end[2]

            # Force reading with simulated noise
            force, _, _, _, _, _, _ = sensing.read(rtde_r,std=std)

            if force_lowpass_alpha:
                filtered_force = lowPassFilter(force,filtered_force,force_lowpass_alpha)
            else:
                filtered_force = force
            force_z_err = filtered_force[2] - desired_z_force

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


            if vac_distance_threshold:
                if z_error < vac_distance_threshold:
                    if np.any(K_fac - 1):
                        K_scale = 1 + (K_fac - 1) * (1 - z_error / vac_distance_threshold)
                        for i in range(6):
                            K_upd[i, i] = K[i, i] * K_scale[i]
                    if np.any(C_fac - 1):
                        C_scale = 1 + (C_fac - 1) * (1 - z_error / vac_distance_threshold)
                        for i in range(6):
                            C_upd[i, i] = C[i, i] * C_scale[i]
                    controller.update_matrices(M, C_upd, K_upd)




            state = controller(tau_ext, state)
            offset = state[:6]       # position offset computed by the controller
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
            data = dict(zip(csv_fields,val))

            if csv_writer:
                csv_writer.writerow(data)

            rtde_c.servoL(target_pose.tolist(), 0.1, 0.5, dt, 0.03, 300)
            rtde_c.waitPeriod(t_start)
        rtde_c.servoStop()

def lowPassFilter(new_value, prev_filtered, alpha):
    """
    Exponential moving average (EMA) low-pass filter for vector data.

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
    sleep(post_move_wait)
    return data

def initializeConnections(robot_ip, freq, hec_path, out_dir, serial_device = None, sertimeout = 1, gamma_noise_rate = 0):

    # Connect to gripper (Arduino)
    ser = gripper.connectGripper(serial_device, sertimeout)

    # Connect to UR robot
    print("Connecting to UR RTDE receive and control interfaces... ",end="")
    rtde_c = RTDEControlInterface(robot_ip, freq)
    rtde_r = RTDEReceiveInterface(robot_ip, freq)
    print("OK")

    # Connect to camera
    print("Connecting to RealSense camera... ",end="")
    camera = RealSenseInterface(hec_path,out_dir,snow_rate=gamma_noise_rate)
    print("OK")

    return ser, rtde_c, rtde_r, camera

def getRandomViewQ(rtde_c, viewP, viewQ, spread, log_path=None):
    while True:
        randPose, xyz = getRandomPoseXYZ(viewP, spread)
        print(xyz)
        if rtde_c.getInverseKinematicsHasSolution(randPose):
            break
    randQ = rtde_c.getInverseKinematics(randPose, qnear=viewQ)

    msg = f"View pose: {randPose.tolist()} (X-Y-Z offset: {xyz.tolist()})"
    if log_path:
        with log_path.open("a") as f:
            f.write(msg + "\n")
    print(msg)
    urMoveJ(rtde_c, rtde_c.getInverseKinematics(randPose,qnear=viewQ))
    return randQ

def getRandomPoseXYZ(pose, spread):
    rand_gen = np.random.default_rng()
    # normal distribution with 99.7% of values within spread
    xyz = rand_gen.normal(loc=0,scale=spread/3,size=3)
    pose_delta = np.concatenate([xyz,np.zeros(3)])
    randPose = pose + pose_delta
    return randPose, xyz

def detectInterface(camera, detector, rtde_r, spread=0.0, interface_type="big_interface", attempts=3,
                    detection_save_path = None, depth_save_path = None, img_save_path=None, log_path=None):

    print("Detecting interface")

    for i in range(attempts):
        print(f"Attempt {i}")
        color_image, depth_image, depth_colormap = camera.get_frames()

        if detection_save_path:
            det_i_save_path = detection_save_path.with_stem(detection_save_path.stem + f"_{i}")
        else:
            det_i_save_path = None
        if img_save_path:
            img_i_save_path = img_save_path.with_stem(img_save_path.stem + f"_{i}")
        else:
            img_i_save_path = None
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
        else:
            save_image = None

        align_pose = None
        for obj in detections:

            x, y, w, h = (round(i) for i in obj["bbox"])
            cx, cy = (round(i) for i in obj["center"])

            baseTee_matrix = sensing.getPoseMatrix(rtde_r)

            depth = obj["depth"]

            obj_coords_base, obj_coords_cam, obj_coords_ee = transform_to_robot_frame(obj["center"], depth, baseTee_matrix,camera.camera_matrix,camera.T_cam_matrix)
            obj_x, obj_y, obj_z = obj_coords_base  # Extract X, Y, Z

            if display_image.size:
                cv2.rectangle(display_image, (x, y), (x + w, y + h), [0,255,0], 2)
                cv2.putText(display_image, f"Class: {obj["class_id"]} | ({obj_x:.3f}, {obj_y:.3f}, {obj_z:.3f}) | Depth: {depth:.3f}",
                            (x, y + h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,255,0], 2)


            if save_image:
                cv2.rectangle(save_image, (x, y), (x + w, y + h), [0,255,0], 2)
                cv2.putText(save_image, f"Class: {obj["class_id"]} | ({obj_x:.3f}, {obj_y:.3f}, {obj_z:.3f})",
                            (x, y + h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,255,0], 2)

            # TODO: is this correct? The last matching detection is chosen?
            if match and obj["label"] == interface_type:
                if display_image.size:
                    cv2.circle(display_image,obj["depth_spot"], 3, [0,255,0],-1) # negative thickness = filled
                align_pose = np.array([obj_x, obj_y, 0.36, 0, math.pi, 0])

        if display_image.size:
            Image.fromarray(display_image[:, :, ::-1]).show()

        if save_image and det_i_save_path:
            cv2.imwrite(det_i_save_path, save_image)


        # cv2.imshow("Interface Detections", display_image)
        # cv2.waitKey(1)

        if align_pose:
            if spread:
                align_pose, xyz_offset = getRandomPoseXYZ(align_pose, spread)
            else:
                xyz_offset = np.zeros(3)
            insert_pose = align_pose.copy()
            insert_pose[2] = 0.216
            msg = f"Alignment pose: {align_pose.tolist()} (X-Y offset: {xyz_offset.tolist()})\nInsertion pose: {insert_pose.tolist()}"
            print(msg)
            if log_path:
                with log_path.open("a") as f:
                    f.write(msg+"\n")
            return align_pose, insert_pose

    raise Exception(f"No detections after {attempts} attempts")

class EvalPrefix(StrEnum):
    SCENE  = "based on the image, evaluate the whole interface is visible or not, then plan the robotic actions for gripper engagement of transporter."
    ALIGNMENT  = "based on the image, and the distance to interface {ultrasonic_dis_cm}, evaluate if the alignment between the gripper and interface is good or not. then plan the following actions for engagement."
    INSERTION  = "based on the image, contact wrench {tcp_wrench_N_Nm}, and the distance to interface {ultrasonic_dis_cm}, evaluate if the insertion between the gripper and interface is good or not. then plan the following actions for engagement."
    ENGAGEMENT = "based on the image, the distance to interface {ultrasonic_dis_cm}, and the folding arm force {force_gauge}, evaluate the engagement between the folding arm and interface is good or not. then plan the following actions for engagement."

def stringify_wrench(w) -> str:
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

def evaluateScene(model, camera, eval_mode, img_save_path=None, log_path=None):
    succ_actions = " planned actions: detect interface; align gripper with interface; evaluate alignment; insert gripper; evaluate insertion; engage gripper; evaluate engagement."

    response = ""
    msg = ""
    succ = True
    if eval_mode[0] == 2:
        prefix = EvalPrefix.SCENE
        response = evaluate(model, camera, prefix, img_save_path=img_save_path, log_path=log_path)
    else:
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

def evaluateAlignment(model, camera, ard, rtde_r, eval_mode, N, dt, std=0.0, img_save_path=None, log_path=None):
    force_mean, dist_mean, arms_mean, _, _, _, _ = sensing.read(rtde_r,ard=ard,N=N,dt=dt,std=std)

    dist_str = stringify_distance(dist_mean)

    succ_actions = " planned actions: insert gripper; evaluate insertion; engage gripper; evaluate engagement."
    fail_actions = " planned actions: chase interface; detect interface; align gripper with interface; evaluate alignment."

    response = ""
    msg = ""
    succ = True
    if eval_mode[1] == 2:
        prefix = EvalPrefix.ALIGNMENT.format(ultrasonic_dis_cm=dist_str)
        response = evaluate(model, camera, prefix, img_save_path=img_save_path, log_path=log_path)
    else:
        if eval_mode[1] == 0:
            msg = "No evaluation."
        elif eval_mode[1] == 1:
            succ = dist_mean <= (22-3)
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

def evaluateInsertion(model, camera, rtde_r, ard, eval_mode, N, dt, std=0.0, img_save_path=None, log_path=None):
    force_mean, dist_mean, arms_mean, _, _, _, _ = sensing.read(rtde_r,ard=ard,N=N,dt=dt,std=std)
    force_str = stringify_wrench(force_mean)
    dist_str = stringify_distance(dist_mean)

    succ_actions = " planned actions: engage gripper; evaluate engagement."
    fail_actions = " planned actions: remove gripper; insert gripper; evaluate insertion."

    response = ""
    succ = True
    msg = ""
    if eval_mode[2] == 2:
        prefix = EvalPrefix.INSERTION.format(tcp_wrench_N_Nm = force_str, ultrasonic_dis_cm = dist_str)
        response = evaluate(model, camera, prefix, img_save_path = img_save_path, log_path = log_path)
    else:
        if eval_mode[2] == 0:
            msg = "No evaluation."
        elif eval_mode[2] == 1:
            eval1 = (dist_mean > 3) and (dist_mean < 5)
            eval2 = force_mean[2] >= 4.0
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

def evaluateEngagement(model, camera, ard, rtde_r, eval_mode, N, dt, std=0.0, img_save_path=None, log_path=None):
    force_mean, dist_mean, arms_mean, _, _, _, _ = sensing.read(rtde_r,ard=ard,N=N,dt=dt,std=std)
    arms_str = stringify_force_gauge(arms_mean)
    dist_str = stringify_distance(dist_mean)

    succ_actions = " planned actions: finished."
    fail_actions = " planned actions: disengage gripper; engage gripper; evaluate engagement."

    response = ""
    succ = True
    msg = ""
    if eval_mode[3] == 2:
        prefix = EvalPrefix.ENGAGEMENT.format(force_gauge = arms_str, ultrasonic_dis_cm = dist_str)
        response = evaluate(model, camera, prefix, img_save_path = img_save_path, log_path = log_path)
    else:
        if eval_mode[3] == 0:
            msg = "No evaluation."
        if eval_mode[3] == 1:
            eval1 = np.any(np.array(arms_mean) >= 4.0)
            eval2 = (dist_mean > 3) and (dist_mean < 5)
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


def evaluate(model, camera, prefix, img_save_path = None, log_path = None):
    img_rgb, _, _ = camera.get_frames()
    model.setMode("vqa")
    response, _ = model.infer(img_rgb, prefix, img_save_path = img_save_path, log_path = log_path)
    return response

def finalEvaluation(rtde_r, ard, eval_mode, now, spread, N, dt, std=0.0, csv_path = None):
    force_mean, dist_mean, arms_mean, _, _, _, _ = sensing.read(rtde_r,ard=ard,N=N,dt=dt,std=std)
    # force_str = stringify_wrench(force_mean)
    dist_str = stringify_distance(dist_mean)
    arms_str = stringify_force_gauge(arms_mean)

    eval1 = np.any(np.array(arms_mean) >= 4.0)
    eval2 = (dist_mean > 3) and (dist_mean < 5)

    succ = eval1 and eval2
    msg = f"Final heuristic evaluation: distance: {dist_str}, arm forces: {arms_str}.\nFinal success: {succ}."
    print(msg)
    if csv_path:
        data = [now, "YOLO"]
        data.extend(eval_mode)
        data.append(spread)
        data.append(dist_mean)
        data.extend(arms_mean)
        data.extend(force_mean)
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
                # status = status_actionplan[0]
                actionplan = status_actionplan[1].split("; ")
                if actionplan != status_actionplan[1]:
                    actionplan = deque(actionplan)
                    continue
            raise Exception("Failed to parse response")

        if actionplan:  # pyright: ignore[reportPossiblyUnboundVariable]
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
        "evaluate scene":               "response, response_id = evaluateScene(evaluator_model, camera, eval_mode, img_save_path=rgb_save_path, log_path=log)",
        "chase interface":              "rand_view_q = urMoveJ(rtde_c, getRandomViewQ(rtde_c, viewP, viewQ, spread = rand_spread_scene))",
        "detect interface":             "align_pose, insert_pose = detectInterface(camera, detector_model, rtde_r, spread = rand_spread_align, detection_save_path = detect_save_path, depth_save_path = z_save_path, img_save_path=rgb_save_path, log_path=log)",
        "align gripper with interface": "urMoveJ(rtde_c, align_pose, isIK=True)",
        "evaluate alignment":           "response, response_id = evaluateAlignment(evaluator_model, camera, ser, rtde_r, eval_mode, n_samp, dt_samp, img_save_path=rgb_save_path, log_path=log)",
        "insert gripper":               "variableAdmittanceMoveL(rtde_c, rtde_r, insert_pose, 20.0, dt, admit_M, insert_C, insert_K, K_fac = insert_K_fac, C_fac = insert_C_fac, desired_z_force = insert_Fz, vac_distance_threshold = 0.01",
        "evaluate insertion":           "response, response_id = evaluateInsertion(evaluator_model, camera, rtde_r, ser, eval_mode, n_samp, dt_samp, img_save_path=rgb_save_path, log_path=log)",
        "remove gripper":               "variableAdmittanceMoveL(rtde_c, rtde_r, align_pose, 10.0, dt, admit_M, remove_C, remove_K, zero_ft = False, out_dir=out_dir))",
        "engage gripper":               "engageGripper(ser, True, servo_time)",
        "evaluate engagement":          "response, response_id = evaluateEngagement(evaluator_model, camera, ser, rtde_r, eval_mode, n_samp, dt_samp, img_save_path=rgb_save_path, log_path=log)",
        "disengage gripper":            "engageGripper(ser, False, servo_time)",
        "finished":                     """finalEvaluation(rtde_r, ser, eval_mode, now, (rand_spread_scene, rand_spread_align), n_samp, dt_samp, csv_path = global_csv)
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

    # Amount of simulated gamma radiation
    gamma_dose_rate = 100/60 # Gy/min

    # Returns Poisson exected value rate (DN/s) as a function of dose rate
    gamma_noise_rate = snow.model(gamma_dose_rate)

    # Crudely approximate the noise in other sensors as Gaussian with same
    # normalized variance as the image sensor (mean = variance for Poisson)
    sensor_noise_std_norm = np.sqrt(gamma_noise_rate)/255

    # Number of samples and time between samples when reading F-T, range, and arm sensors
    n_samp = 50
    dt_samp = 0.04

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
    ldict["ser"], ldict["rtde_c"], ldict["rtde_r"], ldict["camera"] = initializeConnections(ldict["robot_ip"], ldict["freq"], ldict["hec_path"], ldict["out_dir"], gamma_noise_rate=gamma_noise_rate)

    # Hard-coded start pose
    ldict["viewQ"] = [1.8504924774169922, -1.4910245326212426, 0.5884845892535608, -0.6688453716090699, -1.5668700377093714, -0.5082209745990198]
    ldict["viewP"] = ldict["rtde_c"].getForwardKinematics(ldict["viewQ"],tcp_offset=np.zeros(6).tolist())

    # Load the object detection model (YOLO)
    ldict["detector_model"] = detectorYOLO(model_weights_path="object_detection/runs/detect/train2/weights/best.pt",confidence_threshold=0.9)

    # Load the evaluator model (PaliGemma)
    ldict["evaluator_model"] = VLM("vlm_training/check_point/od_vqa_fec_25epoch_new_od_img_v2/checkpoint-6960")

    # Start the main loop while accepting keyboard interrupts
    try:
        main(ldict,action_adapter)
    except KeyboardInterrupt:
        print("Received keyboard interrupt, stopping")
    finally:
        ldict["camera"].stop()
