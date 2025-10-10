#!/usr/bin/env python3
# main.py
import time
import numpy as np
from admittance_controller import ComputeAdmittance
from trajectory_planner import linear_trajectory
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface


def main():
    # --- Configuration ---
    ROBOT_IP = "192.168.1.254"  # <-- Replace with your robot's IP address
    dt = 0.01  # Control loop time step (8 ms)
    frequency = 1 / dt

    # --- Initialize RTDE Interfaces ---
    rtde_c = RTDEControlInterface(ROBOT_IP,frequency)
    rtde_r = RTDEReceiveInterface(ROBOT_IP,frequency)

    # --- Define Admittance Controller Parameters ---
    # These matrices must be tuned to your application.
    Md = np.diag([500, 500, 50, 5, 5, 5])  # Virtual inertia matrix
    Cd = np.diag([50, 50, 10, 50, 50, 50])  # Virtual damping matrix
    Kd = np.diag([1000, 1000, 10, 50, 50, 50])  # Virtual stiffness matrix

    # Create the ComputeAdmittance controller.
    # The controller’s state is a 12-element vector:
    #   state = [position_offset (6), velocity (6)]
    admittance_ctrl = ComputeAdmittance(Md, Cd, Kd, dt)
    state = np.zeros(12)  # initial state: zero offset and zero velocity

    # Define desired force. For compliance, this is typically zero.
    desired_force = np.zeros(6)

    # --- Retrieve the Current TCP Pose ---
    # The TCP pose is a 6D vector: [x, y, z, rx, ry, rz]
    nominal_pose = np.array(rtde_r.getActualTCPPose())

    # --- Trajectory Planning ---
    # For example, define a goal pose that is 0.1 m lower in z.
    goal_pose = nominal_pose.copy()
    # goal_pose[2] -= 0.1
    num_points = 100
    trajectory = linear_trajectory(nominal_pose, goal_pose, num_points)
    traj_index = 0
    rtde_c.zeroFtSensor()

    print("Starting admittance control loop. Press Ctrl+C to stop.")

    try:
        while True:
            loop_start = time.time()

            # --- Read Measured Force ---
            # getActualTCPForce() returns a list [Fx, Fy, Fz, Mx, My, Mz]
            measured_force = np.array(rtde_r.getActualTCPForce())
            # Compute force error (if desired_force is nonzero, else it's just the measured force)
            tau_ext = measured_force - desired_force

            # --- Update Admittance Controller State ---
            state = admittance_ctrl(tau_ext, state)
            # The first 6 elements of state correspond to the position offset.
            offset = state[:6]

            # --- Update Nominal Pose Based on Trajectory ---
            if traj_index < len(trajectory):
                nominal_pose = trajectory[traj_index]
                traj_index += 1
            # Else, keep using the last point

            # --- Compute New Target Pose ---
            target_pose = nominal_pose + offset

            # --- Command the Robot ---
            speed = 0.1  # Linear speed in m/s
            acceleration = 0.5  # Acceleration in m/s²
            rtde_c.moveL(target_pose.tolist(), speed, acceleration,True)

            # --- Maintain Control Loop Timing ---
            elapsed = time.time() - loop_start
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\nAdmittance control loop interrupted by user.")

    finally:
        print("Stopping robot and closing connections.")
        rtde_c.stopScript()  # Stop any running robot scripts
        rtde_c.disconnect()
        rtde_r.disconnect()


if __name__ == "__main__":
    main()
