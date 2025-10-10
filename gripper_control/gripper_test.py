#!/usr/bin/env python3
import serial
from serial.tools.list_ports import comports
import platform
import time
from random import randint
from pathlib import Path
import csv
from enum import IntEnum

# UR_RTDE
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

class evaluations(IntEnum):
    NONE = 0
    DETECTION = 1
    APPROACH = 2
    INSERTION = 3
    LOCKING = 4

robot_ip = "192.168.1.254"
control_loop_freq = 100 
control_loop_dt = 1 / control_loop_freq  # Control loop time step (seconds)
control_loop_dt_ns = control_loop_dt*1e9

# Connect to robot
# rtde_c = RTDEControlInterface(robot_ip, control_loop_freq)
rtde_r = RTDEReceiveInterface(robot_ip, control_loop_freq)

# Zero the force-torque sensor before starting.
# rtde_c.zeroFtSensor()

sertimeout = 1
serdev = [i.device for i in comports()]
if platform.system() == "Linux":
    serdev = [i for i in serdev if i.startswith("/dev/ttyACM")]
dev = serdev[0]
print("Connecting to " + dev)
ser = serial.Serial(dev,timeout=sertimeout,baudrate=115200,write_timeout=sertimeout)

dt = 0.01

move_gripper = False

write_sens_csv = False
sens_csv_path = Path("test.csv")
sens_csv_fields = ["Time",
                #    "Pose",
                #    "F-T",
                   "Distance",
                   "Arms",
                   "Evaluation"]

def gripperSerialCmd(ard,msg,enc="ASCII",verbose=False):
    if verbose:
        print("Wrote: {}".format(msg))
    ard.write(bytes(msg,encoding=enc))
    res = ser.readline().decode(encoding='ASCII')
    if verbose:
        if res:
            # Response already includes a newline
            print("Read: {}".format(res),end="")
        else:
            print("No response")
    return res

def getGripperSensors(ard):
    res = gripperSerialCmd(ard,"5")
    if res:
        val = res.split(", ")
        t = int(val[0])
        sens = val[3].split(" ")
        dist = int(sens[0])
        arms = (float(sens[1]), float(sens[2]))
        return t, dist, arms
    else:
        return None, None, None

def main(sens_csv_writer):
    x = 1

    # Make sure the Arduino is responding consistently
    print("Confirming Arduino connection")
    for _ in range(5):
        res = gripperSerialCmd(ser,"0",verbose=True)
    if not res:
        raise(Exception("Arduino not responding"))

    while True:
        # z = randint(0,9)

        # Gripper cmd
        if move_gripper:
            gripperSerialCmd(ser,str(x),verbose=True)
            if x == 1:
                x = 2
            elif x == 2:
                x = 1

        # Read sensors
        t, dist, arms = getGripperSensors(ser)
        pose = rtde_r.getActualTCPPose()
        force = rtde_r.getActualTCPForce()
        sens = dict(zip(sens_csv_fields, [t, dist, arms, evaluations.NONE]))
        # sens = {"Time":t,
        #         "Pose":pose,
        #         "F-T":force,
        #         "Distance":dist,
        #         "Arms":arms,
        #         "Evaluation":evaluations.NONE}

        print(f"{t} {dist} {arms}")
        if write_sens_csv:
            sens_csv_writer.writerow(sens)

        time.sleep(dt)

if __name__ == '__main__':
    if write_sens_csv:
        print(f"Opening CSV file: {sens_csv_path}")
        sens_csv_file = open(sens_csv_path,"w")
        sens_csv_writer = csv.DictWriter(sens_csv_file,sens_csv_fields,delimiter=";")
        sens_csv_writer.writeheader()
    else:
        sens_csv_writer = None

    try:
        main(sens_csv_writer=sens_csv_writer)
    except KeyboardInterrupt:
        print("Received keyboard interrupt, stopping")
    finally:
        if write_sens_csv:
            sens_csv_file.close()
            print("Reading csv file")
            with open(sens_csv_path) as sens_csv_file:
                csv_reader = csv.DictReader(sens_csv_file,delimiter=";")
                for i,row in enumerate(csv_reader):
                    print(f"Row {i}:{row}")
        

