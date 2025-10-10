#!/usr/bin/env python3
import time
import numpy as np
import matplotlib.pyplot as plt
from admittance_controller import ComputeAdmittance
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

def low_pass_filter(new_value, prev_filtered, alpha):
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

# def main():
# --- Configuration ---
ROBOT_IP = "192.168.1.254"  # <-- Replace with your robot's IP address
dt = 0.01                   # Control loop time step (seconds)
frequency = 1 / dt
alpha = 0.2                 # Low-pass filter coefficient

# --- Initialize RTDE Interfaces ---
rtde_c = RTDEControlInterface(ROBOT_IP, frequency)
rtde_r = RTDEReceiveInterface(ROBOT_IP, frequency)

# Zero the force-torque sensor before starting.
rtde_c.zeroFtSensor()

# --- Admittance Controller Parameters ---
# Tune these matrices for your application.
Md = np.diag([50, 50, 50, 50, 50, 50])  # Virtual inertia matrix
Cd = np.diag([2000, 2000, 1000, 5000, 5000, 5000])  # Virtual damping matrix
Kd = np.diag([0, 0, 500, 5000, 5000, 5000])  # Virtual stiffness matrix

# Create the ComputeAdmittance controller.
# The state is a 12-element vector: [position_offset (6), velocity (6)]
admittance_ctrl = ComputeAdmittance(Md, Cd, Kd, dt)
state = np.zeros(12)  # initial state: zero offset and zero velocity

# Define desired force (typically zero for pure compliance)
desired_force = np.zeros(6)

# --- Define the Reference Trajectory ---
# Get the current TCP pose as the starting reference.
initial_pose = np.array(rtde_r.getActualTCPPose())  # [x, y, z, rx, ry, rz]
# Define a goal pose that is 20 cm lower in the z axis.
goal_pose = np.copy(initial_pose)
goal_pose[2] -= 0.2   # Move down by 20 cm

# Define the total time over which the trajectory should be executed.
T_traj = 20.0  # seconds

# --- Initialize the filtered force (6D) to zeros.
filtered_force = np.zeros(6)

# --- Data Logging Variables ---
time_log = []
raw_force_log = []       # raw force measurements [Fx, Fy, Fz, Mx, My, Mz]
filtered_force_log = []  # filtered force measurements
offset_log = []          # admittance-computed position offset (error)
target_pose_log = []     # robot target position (reference + offset)
velocity_log = []        # computed velocity (state[6:12])
reference_pose_log = []  # reference trajectory (computed from initial & goal)

# --- Enable Interactive Mode so that figures remain visible ---
plt.ion()

print("Starting servoL control loop with trajectory and logging. Press Ctrl+C to stop.")
start_time = time.time()
iteration = 0
try:
    # Ensure any previous servo commands are stopped.
    rtde_c.servoStop()
    while True:
        loop_start = time.time()
        t = loop_start - start_time
        time_log.append(t)

        # --- Generate the Reference Trajectory ---
        # Interpolate between the initial and goal pose.
        if t < T_traj:
            ratio = t / T_traj
        else:
            ratio = 1.0
        reference_pose = initial_pose + ratio * (goal_pose - initial_pose)
        reference_pose_log.append(reference_pose.copy())

        # --- Read Measured Force ---
        measured_force = np.array(rtde_r.getActualTCPForce())
        raw_force_log.append(measured_force.copy())

        # --- Apply Low-Pass Filtering ---
        filtered_force = low_pass_filter(measured_force, filtered_force, alpha)
        filtered_force_log.append(filtered_force.copy())

        # --- Compute External Force Error ---
        tau_ext = filtered_force - desired_force

        # --- Update Admittance Controller State ---
        state = admittance_ctrl(tau_ext, state)
        offset = state[:6]       # position offset computed by the controller
        velocity = state[6:12]   # computed velocity (the second half)
        offset_log.append(offset.copy())
        velocity_log.append(velocity.copy())

        # --- Compute the New Target Pose ---
        # Here we add the computed admittance offset to the reference trajectory.
        target_pose = reference_pose + offset
        target_pose_log.append(target_pose.copy())

        # --- Command the Robot Using servoL ---
        rtde_c.servoL(target_pose.tolist(), 0.5, 0.5, 0.01, 0.03, 300)

        # --- (Optional) Print Variables Every 100 Iterations ---
        # iteration += 1
        # if iteration % 100 == 0:
        #     print(f"t = {t:.2f} s")
        #     print(f"  Measured Force: {measured_force}")
        #     print(f"  Filtered Force: {filtered_force}")
        #     print(f"  Offset: {offset}")
        #     print(f"  Reference Pose: {reference_pose}")
        #     print(f"  Target Pose: {target_pose}")
        #     print(f"  Computed Velocity: {velocity}\n")

        # --- Maintain Control Loop Timing ---
        elapsed = time.time() - loop_start
        if dt - elapsed > 0:
            time.sleep(dt - elapsed)

except KeyboardInterrupt:
    print("\nServoL control loop interrupted by user.")
finally:
    print("Stopping servoL and closing connections.")
    rtde_c.servoStop()    # Stop servo motion if applicable
    rtde_c.stopScript()   # Stop any running scripts on the robot
    rtde_c.disconnect()
    rtde_r.disconnect()

    # Convert logged data to numpy arrays.
    time_log = np.array(time_log)
    raw_force_log = np.array(raw_force_log)           # shape: (n_samples, 6)
    filtered_force_log = np.array(filtered_force_log)   # shape: (n_samples, 6)
    offset_log = np.array(offset_log)                   # shape: (n_samples, 6)
    target_pose_log = np.array(target_pose_log)         # shape: (n_samples, 6)
    velocity_log = np.array(velocity_log)               # shape: (n_samples, 6)
    reference_pose_log = np.array(reference_pose_log)   # shape: (n_samples, 6)

    # --- Plot Figures ---

    # Figure 1: Raw Force vs. Filtered Force
    force_labels = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
    fig1, axs1 = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
    for i in range(6):
        axs1[i].plot(time_log, raw_force_log[:, i], label="Raw " + force_labels[i],
                     color="tab:blue", alpha=0.7)
        axs1[i].plot(time_log, filtered_force_log[:, i], label="Filtered " + force_labels[i],
                     color="tab:red", linewidth=2)
        axs1[i].set_ylabel(force_labels[i])
        axs1[i].legend(loc='upper right')
        axs1[i].grid(True)
    axs1[-1].set_xlabel("Time (s)")
    fig1.suptitle("Raw vs. Filtered Force Measurements")
    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show(block=True)

    # Figure 2: Robot Target Position and Position Error (Offset)
    fig2, axs2 = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
    for i in range(6):
        axs2[i].plot(time_log, target_pose_log[:, i], label="Target Position " + force_labels[i],
                     color="tab:green")
        axs2[i].plot(time_log, reference_pose_log[:, i], '--', label="Reference " + force_labels[i],
                     color="tab:orange")
        axs2[i].plot(time_log, offset_log[:, i], label="Offset " + force_labels[i],
                     color="tab:purple", linewidth=2)
        axs2[i].set_ylabel(force_labels[i])
        axs2[i].legend(loc='upper right')
        axs2[i].grid(True)
    axs2[-1].set_xlabel("Time (s)")
    fig2.suptitle("Robot Target Position and Position Error (Offset)")
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show(block=True)

    # # Figure 3: Robot Target (Computed) Velocity and Velocity Error
    # fig3, axs3 = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
    # for i in range(6):
    #     axs3[i].plot(time_log, velocity_log[:, i], label="Computed Velocity " + force_labels[i],
    #                  color="tab:cyan", linewidth=2)
    #     axs3[i].set_ylabel(force_labels[i])
    #     axs3[i].legend(loc='upper right')
    #     axs3[i].grid(True)
    # axs3[-1].set_xlabel("Time (s)")
    # fig3.suptitle("Robot Target (Computed) Velocity and Velocity Error")
    # fig3.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show(block=True)
    #
    # # Figure 4: Robot Reference Trajectory (No External Force)
    # fig4, axs4 = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
    # for i in range(6):
    #     axs4[i].plot(time_log, reference_pose_log[:, i], label="Reference Trajectory " + force_labels[i],
    #                  color="tab:cyan")
    #     axs4[i].set_ylabel(force_labels[i])
    #     axs4[i].legend(loc='upper right')
    #     axs4[i].grid(True)
    # axs4[-1].set_xlabel("Time (s)")
    # fig4.suptitle("Robot Reference Trajectory (No External Force)")
    # fig4.tight_layout(rect=[0, 0.03, 1, 0.95])
    #
    # # Use blocking show so that all figure windows remain open.
    # plt.show(block=True)

# if __name__ == "__main__":
#     main()
