#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# --- Read CSV Data ---
# This assumes that the CSV file "logged_data.csv" is in the same directory.
# The CSV is assumed to have the header:
# time,raw_Fx,raw_Fy,raw_Fz,raw_Mx,raw_My,raw_Mz,
# filtered_Fx,filtered_Fy,filtered_Fz,filtered_Mx,filtered_My,filtered_Mz,
# offset_Fx,offset_Fy,offset_Fz,offset_Mx,offset_My,offset_Mz,
# target_Fx,target_Fy,target_Fz,target_Mx,target_My,target_Mz,
# velocity_Fx,velocity_Fy,velocity_Fz,velocity_Mx,velocity_My,velocity_Mz,
# ref_Fx,ref_Fy,ref_Fz,ref_Mx,ref_My,ref_Mz

# Use np.genfromtxt to read the CSV, skipping the header line.
data = np.genfromtxt('logged_data.csv', delimiter=',', skip_header=1)

# Extract columns:
# time_log: column 0
time_log = data[:, 0]

# Raw force: columns 1-3 (Fx, Fy, Fz)
raw_force_log = data[:, 1:4]

# Filtered force: columns 7-9
filtered_force_log = data[:, 7:10]

# Offset (position): columns 13-15 (offset_Fx, offset_Fy, offset_Fz)
offset_log = data[:, 13:16]

# Target position: columns 19-21 (target_Fx, target_Fy, target_Fz)
target_pose_log = data[:, 19:22]

# Reference position: columns 31-33 (ref_Fx, ref_Fy, ref_Fz)
reference_pose_log = data[:, 31:34]

# (Note: Adjust column indices if your CSV layout is different.)

# --- Convert position data from meters to millimeters ---
offset_log_mm = offset_log * 1000
target_pose_log_mm = target_pose_log * 1000
reference_pose_log_mm = reference_pose_log * 1000

# --- Set Global Font Properties (without LaTeX) ---
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['font.size'] = 7

# --- Create a 3x2 Figure ---
fig, axs = plt.subplots(3, 2, figsize=(4.13, 2.5), dpi=800, sharex=True)

# --- Define Labels for Axes ---
axis_labels = ['X', 'Y', 'Z']
force_labels = ['Fx', 'Fy', 'Fz']

# --- Left Column: Force Plots (for X, Y, Z) ---
for i in range(3):
    ax = axs[i, 0]
    ax.plot(time_log, raw_force_log[:, i], 'k-')
    # ax.plot(time_log, filtered_force_log[:, i], 'r--', label='Filt ' + force_labels[i])
    ax.set_ylabel(f"{force_labels[i]} (N)")
    # ax.set_title(force_labels[i])
    # ax.legend(loc='upper right', frameon=False)
    ax.grid(True)

# --- Right Column: Motion (Position) Plots ---
for i in range(3):
    ax = axs[i, 1]
    ax.plot(time_log, offset_log_mm[:, i], 'b-', label='Offset ' + axis_labels[i])
    ax.plot(time_log, target_pose_log_mm[:, i], 'r-', label='Compliant ' + axis_labels[i])
    ax.plot(time_log, reference_pose_log_mm[:, i], 'k--', label='Reference ' + axis_labels[i])
    ax.set_ylabel(f"{axis_labels[i]} (mm)")
    # ax.set_title(axis_labels[i])
    ax.legend(loc='upper right', frameon=False)
    ax.grid(True)

# --- Set the X-axis label for the bottom subplots ---
axs[2, 0].set_xlabel("Time (s)")
axs[2, 1].set_xlabel("Time (s)")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
# --- Display the Figure ---
plt.show(block=True)
fig.savefig('VAC_for_gripper_insertion_v2.png', dpi=800, bbox_inches='tight')

