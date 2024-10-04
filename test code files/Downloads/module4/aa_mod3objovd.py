#!/usr/bin/env python

from easygopigo3 import EasyGoPiGo3
from gopigo3 import FirmwareVersionError
import sys
import signal
from time import sleep

DEBUG = False 
MAX_DISTANCE = 2300
MIN_DISTANCE = 150 
NO_OBSTACLE = 3000
ERROR = 0 
MAX_SPEED = 300 
MIN_SPEED = 100 

robot_operating = True

def signal_handler(signal, frame):
    global robot_operating
    print("CTRL-C combination pressed")
    robot_operating = False

def debug(string):
    if DEBUG is True:
        print("Debug: " + str(string))

def Main():
    robot_operating = True
    try:
        gopigo3 = EasyGoPiGo3()
        distance_sensor = gopigo3.init_distance_sensor()

    except IOError as msg:
        debug(msg)
        sys.exit(1)

    except Exception as msg:
        debug(msg)
        sys.exit(1)

#    gopigo3.set_speed(200)
#    gopigo3.drive_cm(50,blocking = False)
#    gopigo3.turn_degrees(90)
    timeout = 0
    
    while robot_operating:
        current_degree = gopigo3.read_encoders()
        current_degree_avrg = gopigo3.read_encoders_average()
        current_distance = distance_sensor.read_mm()
        print("Wheel 1 Degree : {:4}  Wheel 2 Degree : {:4} Encoder Average : {:4}".format(current_degree[0],current_degree[1],current_degree_avrg))
        print("Current distance : {:4} mm".format(current_distance))
        sleep(0.01)
        if current_distance < 50:
            gopigo3.turn_degrees(180)
         
    gopigo3.stop()


"""    gopigo3_stationary = True
    global robot_operating
    

    while robot_operating:
        current_distance = distance_sensor.read_mm()
        determined_speed = 0

        if current_distance == ERROR:
            print("Cannot reach DistanceSensor. Stopping the process.")
            robot_operating = False    
         elif current_distance < MIN_DISTANCE:
            gopigo3_stationary = True
            gopigo3.stop() 
         else:
            gopigo3_stationary = False
            if current_distance == NO_OBSTACLE:
                determined_speed = MAX_SPEED
            else:
                percent_speed = float(current_distance - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)
                determined_speed = MIN_SPEED + (MAX_SPEED - MIN_SPEED) * percent_speed

            gopigo3.set_speed(determined_speed)
            gopigo3.forward()

        print("Current distance : {:4} mm Current speed: {:4} Stopped: {}".format(current_distance, int(determined_speed), gopigo3_stationary is True ))
        sleep(0.08)
"""

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    Main()
