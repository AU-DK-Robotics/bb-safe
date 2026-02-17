from serial import Serial
from serial.tools.list_ports import comports
import platform
import numpy as np
import time

def connectGripper(serial_device, sertimeout):
    # If no device given, pick one automatically
    if not serial_device:
        serdev = [i.device for i in comports()]
        if platform.system() == "Linux":
            serdev = [i for i in serdev if i.startswith("/dev/ttyACM")]
    else:
        serdev = None

    if serdev:
        serial_device = serdev[0]
    else:
        raise Exception("Could not find any Arduino serial devices")

    print("Connecting to gripper (serial device " + serial_device + ")... ",end="")

    ser = Serial(serial_device,timeout=sertimeout,baudrate=115200,write_timeout=sertimeout)

    # Make sure the Arduino is responding consistently
    res = None
    for _ in range(5):
        res = gripperSerialCmd(ser,"0",verbose=False)
    if not res:
        raise(Exception("Arduino not responding"))
    print("OK")

    return ser

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

def getGripperSensors(ard,rtde_r,T_base_tcp):
    res = gripperSerialCmd(ard,"5")
    if res:
        val = res.split(", ")
        t = int(val[0])
        sens = val[3].split(" ")
        dist = int(sens[0])

        # Apply invisible boundary representing the top of the breeding blanket
        # T_base_tcp = getPoseMatrix(rtde_r)
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
            width_x = 0.197
            width_y = 0.13702
            c = 0.036   # corner chamfer

            # ultrasound coords relative to interface
            x = T_interface_ultra[0,3]
            y = T_interface_ultra[1,3]
            z = T_interface_ultra[2,3]

            # print(T_base_ultra)

            bounds = [ np.abs(y) > width_y/2,
                    np.abs(x) > width_x/2,
                    y > -x + width_x/2 - c + width_y/2,
                    y >  x + width_x/2 - c + width_y/2,
                    y < -x - width_x/2 + c - width_y/2,
                    y <  x - width_x/2 + c - width_y/2 ]

            # Assume vertical gripper orientation
            if np.any(bounds):
                print(f"Restricting distance measurement due to out of bounds: {[int(i) for i in bounds]}")
                dist = round(z*100)

        arms = (float(sens[1]), float(sens[2])) # R, width_x
        return t, dist, arms
    else:
        # return None, None, None
        raise(Exception("Communication with gripper sensors failed"))

def engageGripper(ard,cmd,movetime,post_move_wait=1):
    if cmd:
        gripperSerialCmd(ard,"2")
    else:
        gripperSerialCmd(ard,"1")
    time.sleep(movetime)
    time.sleep(post_move_wait)
