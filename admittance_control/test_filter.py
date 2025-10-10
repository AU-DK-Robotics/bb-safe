#!/usr/bin/env python3
import time
import numpy as np
import matplotlib.pyplot as plt
from rtde_receive import RTDEReceiveInterface
from rtde_control import RTDEControlInterface


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


def main():
    # --- Configuration ---
    ROBOT_IP = "192.168.1.254"  # Replace with your robot's actual IP address
    duration = 30.0  # Duration to collect data (seconds)
    dt = 0.01  # Control loop time step (seconds)
    frequency = 1 / dt  # Communication frequency for RTDE
    alpha = 0.1  # Filter coefficient (0 = heavy smoothing, 1 = no filtering)

    # --- Connect to the RTDE Interfaces ---
    rtde_r = RTDEReceiveInterface(ROBOT_IP, frequency)
    rtde_c = RTDEControlInterface(ROBOT_IP, frequency)

    # Zero the force-torque sensor before starting.
    rtde_c.zeroFtSensor()

    # --- Prepare Data Storage ---
    timestamps = []
    raw_data = []  # To store raw 6D force measurements
    filtered_data = []  # To store filtered 6D force measurements

    # Initialize the filtered force (6D vector) to zeros.
    filtered_force = np.zeros(6)

    print("Collecting 6D force measurements for {:.1f} seconds...".format(duration))
    start_time = time.time()
    while time.time() - start_time < duration:
        current_time = time.time() - start_time
        timestamps.append(current_time)

        # Read the 6D force measurement from the robot.
        # Expected format: [Fx, Fy, Fz, Mx, My, Mz]
        force = np.array(rtde_r.getActualTCPForce())
        raw_data.append(force)

        # Apply the low-pass filter to the new measurement.
        filtered_force = low_pass_filter(force, filtered_force, alpha)
        filtered_data.append(filtered_force.copy())

        # Maintain the control loop timing.
        elapsed = time.time() - start_time - current_time
        if dt - elapsed > 0:
            time.sleep(dt - elapsed)

    # Convert lists to numpy arrays for plotting.
    timestamps = np.array(timestamps)
    raw_data = np.array(raw_data)  # Shape: (num_samples, 6)
    filtered_data = np.array(filtered_data)  # Shape: (num_samples, 6)

    # --- Plotting the Data ---
    force_labels = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
    fig, axs = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
    for i in range(6):
        axs[i].plot(timestamps, raw_data[:, i], label="Raw " + force_labels[i],
                    color="tab:blue", alpha=0.7)
        axs[i].plot(timestamps, filtered_data[:, i], label="Filtered " + force_labels[i],
                    color="tab:red", linewidth=2)
        axs[i].set_ylabel(force_labels[i])
        axs[i].legend(loc='upper right')
        axs[i].grid(True)
    axs[-1].set_xlabel("Time (s)")
    plt.suptitle("Comparison of Raw and Filtered 6D Force Measurements")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- Clean Up: Disconnect from the RTDE Interfaces ---
    rtde_r.disconnect()
    rtde_c.stopScript()
    rtde_c.disconnect()


if __name__ == "__main__":
    main()
