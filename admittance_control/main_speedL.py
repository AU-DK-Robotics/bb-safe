#!/usr/bin/env python3
import time
import numpy as np
from admittance_controller import ComputeAdmittance
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

def main():
    # --- Configuration ---
    ROBOT_IP = "192.168.1.254"  # <-- Replace with your robot's IP address
    dt = 0.01                # Control loop time step (8 ms)
    frequency = 1/dt

    # --- Initialize RTDE Interfaces ---
    rtde_c = RTDEControlInterface(ROBOT_IP, frequency)
    rtde_r = RTDEReceiveInterface(ROBOT_IP, frequency)

    # --- Define Admittance Controller Parameters ---
    # Tuning matrices for the admittance controller.
    Md = np.diag([500, 500, 50, 5, 5, 5])  # Virtual inertia matrix
    Cd = np.diag([50, 50, 10, 50, 50, 50])  # Virtual damping matrix
    Kd = np.diag([1000, 1000, 10, 50, 50, 50])  # Virtual stiffness matrix

    # Create the ComputeAdmittance controller.
    # The controller state is a 12-element vector: [offset (6), velocity (6)]
    admittance_ctrl = ComputeAdmittance(Md, Cd, Kd, dt)
    state = np.zeros(12)  # initial state: zero offset and zero velocity

    # Define desired force. For pure compliance, this is typically zero.
    desired_force = np.zeros(6)
    rtde_c.zeroFtSensor()

    print("Starting admittance speed control loop. Press Ctrl+C to stop.")

    try:
        while True:
            loop_start = time.time()

            # --- Read Measured Force ---
            # getActualTCPForce() returns [Fx, Fy, Fz, Mx, My, Mz]
            measured_force = np.array(rtde_r.getActualTCPForce())
            # Compute external force error
            tau_ext = measured_force - desired_force

            # --- Update Admittance Controller State ---
            # The RK4 integration returns a new state vector of 12 elements.
            state = admittance_ctrl(tau_ext, state)
            # Extract the velocity component (elements 6:12)
            velocity_cmd = state[6:12].tolist()

            # --- Command the Robot Using speedL ---
            # The speedL() function typically accepts:
            #   - A 6D velocity vector (in [m/s, m/s, m/s, rad/s, rad/s, rad/s])
            #   - A target acceleration
            #   - A time duration over which the speed is maintained
            acceleration = 0.5  # (m/s² or rad/s²)
            time_duration = dt  # seconds (adjust as needed)
            rtde_c.speedL(velocity_cmd, acceleration, time_duration)

            # --- Maintain Control Loop Timing ---
            elapsed = time.time() - loop_start
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\nSpeed control loop interrupted by user.")

    finally:
        print("Stopping robot and closing connections.")
        rtde_c.stopScript()  # Stop any running robot scripts
        rtde_c.disconnect()
        rtde_r.disconnect()

if __name__ == "__main__":
    main()
