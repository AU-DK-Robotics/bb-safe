import numpy as np
import time
import cv2
from gripper_control import gripper

def read(ur,ard=None,std=0.0,N=1,dt=1):
    dist_arr = np.zeros((N))
    arms_arr = np.zeros((2,N))
    force_arr = np.zeros((6,N))

    # Start keeping track of time
    t_start = time.perf_counter()

    # i from 0 to N-1
    for i in range(0,N):

        if ard:
            # Read the gripper sensors (ultrasonic range and arm forces)
            T_base_mat = getPoseMatrix(ur)
            _ , dist_arr[i], arms_arr[:,i] = gripper.getGripperSensors(ard,ur,T_base_mat)

        # Read the UR's 6-axis F-T sensor
        force_arr[:,i] = ur.getActualTCPForce()


        # Try to take measurements according to the schedule defined
        # by N and dt
        if N > i+1:
            while (time.perf_counter() - t_start) < (dt*(i+1)):
                pass

    if std>0:
        # add noise equivalent to the normalized base camera noise rate
        # by multiplying by each sensor's full scale, then clipping
        rand = np.random.default_rng()
        noise = rand.normal(loc=0,scale=std,size=(9,N))
        dist_arr = np.clip(dist_arr + noise[0,:]*348,min=2,max=350)
        arms_arr = np.clip(arms_arr + noise[1:3,:]*5,min=0,max=5)
        force_arr[0:3,:] = np.clip(force_arr[0:3,:] + noise[3:6,:]*60,min=-30,max=30)
        force_arr[3:6,:] = np.clip(force_arr[3:6,:] + noise[6:9,:]*20,min=-10,max=10)

    if N > 1:
        dist = np.mean(dist_arr,axis=1)
        arms = np.mean(arms_arr,axis=1)
        force = np.mean(force_arr,axis=1)
    else:
        dist = dist_arr.flatten()
        arms = arms_arr.flatten()
        force = force_arr.flatten()

    return force, dist, arms

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
