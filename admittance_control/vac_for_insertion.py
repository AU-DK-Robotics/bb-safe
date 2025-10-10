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
    """
    return alpha * new_value + (1 - alpha) * prev_filtered

# --------------------- Main Control Program ---------------------
# Configuration
ROBOT_IP = "192.168.1.254"  # Replace with your robot's IP address
dt = 0.01                   # Control loop time step (seconds)
frequency = 1 / dt
alpha = 0.2                 # Low-pass filter coefficient

# Initialize RTDE Interfaces
rtde_c = RTDEControlInterface(ROBOT_IP, frequency)
rtde_r = RTDEReceiveInterface(ROBOT_IP, frequency)
rtde_c.zeroFtSensor()

# Admittance Controller Parameters (baseline)
Md = np.diag([50, 50, 50, 50, 50, 50])   # Virtual inertia matrix
Cd = np.diag([2000, 2000, 1000, 10000, 10000, 10000])  # Virtual damping matrix
Kd = np.diag([500, 500, 500, 10000, 10000, 10000])      # Virtual stiffness matrix

# Create the ComputeAdmittance controller.
admittance_ctrl = ComputeAdmittance(Md, Cd, Kd, dt)
state = np.zeros(12)  # Initial state: zero offset and zero velocity

desired_force = np.zeros(6)  # Typically zero for pure compliance

# Define the Reference Trajectory: [ 2.99935568e-01 -2.33080391e-01  2.31241112e-01 -1.45179306e-04,  3.14142618e+00  3.75846985e-05]
initial_pose = np.array([2.99935568e-01, -2.38080391e-01,  2.31241112e-01, -1.45179306e-04,  3.14142618e+00,  3.75846985e-05])
# initial_pose = np.array(rtde_r.getActualTCPPose())  # Starting TCP pose [x, y, z, rx, ry, rz]
rtde_c.moveL(initial_pose.tolist(),0.05,0.1)
time.sleep(3)
# Command a movement of -0.2 m in z:
reference_goal = np.copy(initial_pose)
# reference_goal[2] -= 0.2  # This is the commanded reference goal (e.g., [0.3100, -0.2100, 0.1120, ...])
# The final desired target (taking into account drift) is:
reference_goal = np.array([0.3190, -0.2168, 0.1120, 0, 3.14146, 0])
final_target = np.array([0.3190, -0.2168, 0.1120, 0, 3.14146, 0])
T_traj = 20.0  # Total trajectory execution time (seconds)

# Initialize the filtered force (6D) to zeros.
filtered_force = np.zeros(6)

# --------------------- Data Logging Variables ---------------------
time_log = []              # Timestamps
raw_force_log = []         # Raw force measurements [Fx, Fy, Fz, Mx, My, Mz]
filtered_force_log = []    # Filtered force measurements
offset_log = []            # Admittance-computed position offset (error)
target_pose_log = []       # Robot target position (reference + offset)
velocity_log = []          # Computed velocity (state[6:12])
reference_pose_log = []    # Reference trajectory (computed from initial & goal)

# Enable Interactive Mode so that figures remain visible.
plt.ion()

# --------------------- Parameters for Variable Impedance at the Final Stage ---------------------
distance_threshold = 0.01       # When the reference position is within 2 cm of the final target...
global_stiffness_factor = 5.0     # ...the stiffness will eventually increase by this factor.
global_damping_factor = 3.0       # ...and the damping by this factor.
# (The increase will be gradual according to how close the reference is to the final target.)

print("Starting servoL control loop with VAC (gradually stiffening near target). Press Ctrl+C to stop.")
start_time = time.time()
iteration = 0

try:
    rtde_c.servoStop()  # Stop any previous servo commands.
    while True:
        loop_start = time.time()
        t = loop_start - start_time
        time_log.append(t)
        iteration += 1

        # --- Generate the Reference Trajectory ---
        # Linearly interpolate from initial_pose to the commanded reference_goal.
        ratio = t / T_traj if t < T_traj else 1.0
        reference_pose = initial_pose + ratio * (reference_goal - initial_pose)
        reference_pose_log.append(reference_pose.copy())

        # --- Read Measured Force and Apply Filtering ---
        measured_force = np.array(rtde_r.getActualTCPForce())
        raw_force_log.append(measured_force.copy())
        filtered_force = low_pass_filter(measured_force, filtered_force, alpha)
        filtered_force_log.append(filtered_force.copy())

        # --- Compute External Force Error ---
        tau_ext = filtered_force - desired_force

        # --- Adaptive Impedance Update ---
        # Compute the Euclidean position error between the current reference_pose and the final_target.
        pos_error = np.linalg.norm(reference_pose[:3] - final_target[:3])
        # Start with baseline stiffness and damping.
        K_updated = Kd.copy()
        C_updated = Cd.copy()
        if pos_error < distance_threshold:
            # Calculate a gradual scaling factor:
            # When pos_error equals distance_threshold, scale = 1; when pos_error is 0, scale = global_factor.
            scale = 1 + (global_stiffness_factor - 1) * (1 - pos_error / distance_threshold)
            scale_damping = 1 + (global_damping_factor - 1) * (1 - pos_error / distance_threshold)
            for i in range(3):  # For x, y, and z
                K_updated[i, i] = Kd[i, i] * scale
                C_updated[i, i] = Cd[i, i] * scale_damping
        # Otherwise, use baseline values (scale = 1).

        # Update the admittance controller with the new matrices.
        admittance_ctrl.update_matrices(Md, C_updated, K_updated)

        # --- Update the Admittance Controller State ---
        state = admittance_ctrl(tau_ext, state)
        offset = state[:6]       # Computed position offset
        velocity = state[6:12]   # Computed velocity
        offset_log.append(offset.copy())
        velocity_log.append(velocity.copy())

        # --- Compute the New Target Pose ---
        # The final commanded target is the reference trajectory plus the computed offset.
        target_pose = reference_pose + offset
        target_pose_log.append(target_pose.copy())

        # --- Optionally Print Logging Information Every 100 Iterations ---
        if iteration % 100 == 0:
            pos_err_val = np.linalg.norm(reference_pose[:3] - final_target[:3])
            print(f"Iteration {iteration}: t = {t:.3f} s, pos_error = {pos_err_val:.4f}")
            print(f"  Reference Pose: {reference_pose[:3]}")
            print(f"  Offset: {offset[:3]}")
            print(f"  Target Pose: {target_pose[:3]}")
            print(f"  Filtered Force: {filtered_force}\n")

        # --- Command the Robot Using servoL ---
        rtde_c.servoL(target_pose.tolist(), 0.5, 0.5, 0.01, 0.03, 200)

        # --- Maintain the Control Loop Timing ---
        elapsed = time.time() - loop_start
        if dt - elapsed > 0:
            time.sleep(dt - elapsed)

except KeyboardInterrupt:
    print("\nServoL control loop interrupted by user.")
finally:
    print("Stopping servoL and closing connections.")
    rtde_c.servoStop()
    rtde_c.stopScript()
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

    # --- Plot Figures (optional) ---
    # Figure 1: Raw Force vs. Filtered Force
    force_labels = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
    fig1, axs1 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    for i in range(3):
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
    fig2, axs2 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    for i in range(3):
        axs2[i].plot(time_log, target_pose_log[:, i]*1000, label="Target Position " + force_labels[i],
                     color="tab:green")
        axs2[i].plot(time_log, reference_pose_log[:, i]*1000, '--', label="Reference " + force_labels[i],
                     color="tab:orange")
        axs2[i].plot(time_log, offset_log[:, i]*1000, label="Offset " + force_labels[i],
                     color="tab:purple", linewidth=2)
        axs2[i].set_ylabel(force_labels[i])
        axs2[i].legend(loc='upper right')
        axs2[i].grid(True)
    axs2[-1].set_xlabel("Time (s)")
    fig2.suptitle("Robot Target Position and Position Error (Offset)")
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show(block=True)

# Combine logged data into a single array.
# time_log: shape (N,)
# raw_force_log, filtered_force_log, offset_log, target_pose_log, velocity_log, reference_pose_log: each of shape (N, 6)
data = np.hstack((
    time_log.reshape(-1, 1),  # reshape time_log to (N,1)
    raw_force_log,  # (N, 6)
    filtered_force_log,  # (N, 6)
    offset_log,  # (N, 6)
    target_pose_log,  # (N, 6)
    velocity_log,  # (N, 6)
    reference_pose_log  # (N, 6)
))

# Create a header string with column labels.
header = (
    "time,"
    "raw_Fx,raw_Fy,raw_Fz,raw_Mx,raw_My,raw_Mz,"
    "filtered_Fx,filtered_Fy,filtered_Fz,filtered_Mx,filtered_My,filtered_Mz,"
    "offset_Fx,offset_Fy,offset_Fz,offset_Mx,offset_My,offset_Mz,"
    "target_Fx,target_Fy,target_Fz,target_Mx,target_My,target_Mz,"
    "velocity_Fx,velocity_Fy,velocity_Fz,velocity_Mx,velocity_My,velocity_Mz,"
    "ref_Fx,ref_Fy,ref_Fz,ref_Mx,ref_My,ref_Mz"
)

# Save the combined data to a CSV file.
np.savetxt('logged_data.csv', data, delimiter=',', header=header, comments='')

print("Logged data saved to 'logged_data.csv'")
