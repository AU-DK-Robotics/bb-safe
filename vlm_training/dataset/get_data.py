#!/usr/bin/env python3
"""
Capture RealSense images and UR RTDE robot state into paired files.

- Press SPACE to capture: saves a color image (JPG) and JSON with:
  * TCP pose
  * TCP wrench (forces/torques)
  * Joint positions
"""

import argparse
import json
import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
import rtde_receive  # UR RTDE

import platform

import serial
from serial import Serial
from serial.tools.list_ports import comports
import time
from datetime import datetime, timezone

def gripperSerialCmd(ard,msg,enc="ASCII",verbose=False):
    if verbose:
        print("Wrote: {}".format(msg))
    ard.write(bytes(msg,encoding=enc))
    res = ard.readline().decode(encoding='ASCII')
    if verbose:
        if res:
            print("Read: {}".format(res), end="")
        else:
            print("No response")
    return res

def getGripperSensors(ard):
    res = gripperSerialCmd(ard,"5")
    if res:
        val = res.split(", ")
        t = int(val[0])
        sens = val[3].split(" ")
        dist = sens[0]
        arms = (float(sens[1]), float(sens[2]))
        return t, dist, arms
    else:
        return None, None, None

def ensure_dirs(base_out: Path) -> Tuple[Path, Path]:
    img_dir = base_out / "images"
    meta_dir = base_out / "meta"
    img_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    return img_dir, meta_dir

def next_index(img_dir: Path) -> int:
    max_idx = 0
    for name in img_dir.glob("image_*.jpg"):
        try:
            idx = int(name.stem.split("_")[1])
            max_idx = max(max_idx, idx)
        except (IndexError, ValueError):
            continue
    return max_idx + 1

FONT = cv2.FONT_HERSHEY_SIMPLEX

def _fmt_force_gauge(force_gauge):
    if force_gauge is None:
        return None
    try:
        vals = list(force_gauge)
        if len(vals) == 0:
            return None
        mag = sum(v*v for v in vals) ** 0.5
        head = ", ".join(f"{v:6.2f}" for v in vals[:3])
        more = "" if len(vals) <= 3 else " …"
        return f"[{head}{more}] | |F|={mag:6.2f} N"
    except TypeError:
        return f"{float(force_gauge):.2f} N"

def draw_hud(img, *, tcp_wrench=None, dist_cm=None, force_gauge=None, fps=None):
    """Overlay a translucent HUD with live readings on top-left."""
    overlay = img.copy()
    lines = []
    if tcp_wrench is not None:
        try:
            Fx, Fy, Fz, Tx, Ty, Tz = tcp_wrench
            lines.append(f"TCP Wrench: Fx={Fx:7.2f}N Fy={Fy:7.2f}N Fz={Fz:7.2f}N")
            lines.append(f"            Tx={Tx:7.2f}Nm Ty={Ty:7.2f}Nm Tz={Tz:7.2f}Nm")
        except Exception:
            lines.append(f"TCP Wrench: {tcp_wrench}")
    if dist_cm is not None:
        lines.append(f"Ultrasonic: {dist_cm} cm")
    if force_gauge is not None:
        fg_line = _fmt_force_gauge(force_gauge)
        if fg_line:
            lines.append(f"Force Gauge: {fg_line}")
    if fps is not None:
        lines.append(f"FPS: {fps:4.1f}")

    if not lines:
        return img

    pad = 8
    lh = 20
    text_sizes = [cv2.getTextSize(s, FONT, 0.5, 1)[0][0] for s in lines]
    panel_w = max(320, max(text_sizes) + 2 * pad) if text_sizes else 320
    panel_h = pad * 2 + lh * len(lines)
    x0, y0 = 10, 10

    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), thickness=-1)
    img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

    y = y0 + pad + 14
    for s in lines:
        cv2.putText(img, s, (x0 + pad, y), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += lh
    return img

def main() -> None:
    parser = argparse.ArgumentParser(description="Capture RealSense images with paired UR RTDE snapshots.")
    parser.add_argument("--robot-ip", default="192.168.1.254", help="IP address of the UR controller")
    parser.add_argument("--out", type=Path, default=Path("validation/evaluation_of_engagement/ng"), help="Output directory")
    parser.add_argument("--width", type=int, default=1280, help="Color stream width")
    parser.add_argument("--height", type=int, default=720, help="Color stream height")
    parser.add_argument("--fps", type=int, default=30, help="Color stream FPS")
    args = parser.parse_args()

    img_dir, meta_dir = ensure_dirs(args.out)

    # Serial
    sertimeout = 1
    serdev = [i.device for i in comports()]
    if platform.system() == "Linux":
        serdev = [i for i in serdev if i.startswith("/dev/ttyACM")]
    dev = serdev[0]
    print("Connecting to serial device " + dev)
    ser = serial.Serial(dev, timeout=sertimeout, baudrate=115200, write_timeout=sertimeout)

    print("Confirming Arduino connection")
    for _ in range(5):
        res = gripperSerialCmd(ser, "0", verbose=True)
    if not res:
        raise Exception("Arduino not responding")

    # UR RTDE
    try:
        rtde_r = rtde_receive.RTDEReceiveInterface(args.robot_ip)
        _ = rtde_r.getTimestamp()
        print(f"✅ Connected to UR RTDE at {args.robot_ip}")
    except Exception as e:
        print(f"❌ Could not connect to UR RTDE at {args.robot_ip}: {e}")
        return

    # RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    try:
        profile = pipeline.start(config)
    except Exception as e:
        print(f"❌ Failed to start RealSense pipeline: {e}")
        return

    print("📸 Press SPACE to capture, 'q' to quit.")

    object_location = [0.3215075438960016, -0.2090870087523362, 0.21679121706309645, -0.004789495997419717, -3.1376360139078385, -0.0031660587154650377]

    idx = next_index(img_dir)
    last_t = time.time()
    ema_fps = None

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Live reads for HUD
            tcp_pose = None
            tcp_wrench = None
            dist = None
            force_gauge = None
            try:
                tcp_pose = rtde_r.getActualTCPPose()
                tcp_wrench = rtde_r.getActualTCPForce()
            except Exception:
                pass
            try:
                _, dist, force_gauge = getGripperSensors(ser)  # dist string -> shown as-is on HUD
            except Exception:
                pass

            color_image = np.asanyarray(color_frame.get_data())

            # FPS (EMA)
            now = time.time()
            dt = now - last_t
            last_t = now
            inst_fps = (1.0 / dt) if dt > 0 else 0.0
            ema_fps = inst_fps if ema_fps is None else (0.9 * ema_fps + 0.1 * inst_fps)

            # HUD
            hud_frame = draw_hud(
                color_image.copy(),
                tcp_wrench=tcp_wrench,
                dist_cm=dist,
                force_gauge=force_gauge,
                fps=ema_fps
            )

            cv2.imshow("RealSense - Press SPACE to Capture, Q to Quit", hud_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if key == ord(" "):  # Capture snapshot + payload (your original structure)
                base = f"image_{idx:03d}"
                img_path = img_dir / f"{base}.jpg"
                json_path = meta_dir / f"{base}.json"

                try:
                    tcp_pose_cap = rtde_r.getActualTCPPose()
                    tcp_wrench_cap = rtde_r.getActualTCPForce()
                    joint_q = rtde_r.getActualQ()
                    misalignment = [a - b for a, b in zip(tcp_pose_cap[:3], object_location[:3])]
                    _, dist_cap, force_gauge_cap = getGripperSensors(ser)
                except Exception as e:
                    print(f"❌ Failed to read UR data: {e}")
                    continue

                if not cv2.imwrite(str(img_path), color_image):
                    print(f"❌ Failed to save image to {img_path}")
                    continue

                ts_utc = datetime.now(timezone.utc).isoformat()
                h, w = color_image.shape[:2]

                # === payload restored ===
                payload = {
                    "timestamp_utc": ts_utc,
                    "image": {
                        "file": str(img_path.relative_to(args.out)),
                        "width": int(w),
                        "height": int(h),
                        "format": "jpg",
                        "source": "realsense-color-bgr8",
                        "frame_timestamp_ms": float(color_frame.get_timestamp()),
                        "frame_number": int(color_frame.get_frame_number()),
                    },
                    "ur": {
                        "tcp_pose_m_rad": [float(v) for v in tcp_pose_cap],
                        "tcp_wrench_N_Nm": [float(v) for v in tcp_wrench_cap],
                        "joint_positions_rad": [float(v) for v in joint_q],
                    },
                    "gripper": {
                        # your upstream returns string "cm"; ensure float for JSON
                        "ultrasonic_dis_cm": float(dist_cap) if dist_cap is not None else None,
                        "force_gauge": [float(v) for v in force_gauge_cap] if force_gauge_cap is not None else None,
                        "misalignment": [float(v) for v in misalignment],
                    },
                    "notes": "Pose is in base frame. Force is generalized TCP wrench. Units per ur_rtde docs.",
                }

                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)

                print(f"✅ Saved: {img_path} and {json_path}")
                idx += 1

    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
