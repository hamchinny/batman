ERROR = 0 # the error that's returned when the DistanceSensor is not found
#!/usr/bin/env python

"""
GoPiGo3 for the Raspberry Pi: an open source robotics platform for the Raspberry Pi.
Copyright (C) 2017  Dexter Industries

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/gpl-3.0.txt>.
"""


from easygopigo3 import EasyGoPiGo3
from gopigo3 import FirmwareVersionError
import sys
import signal
from time import sleep
import queue
import signal
import threading
from math import *
from statistics import mean
from time import sleep

import numpy as np
from curtsies import Input
from di_sensors import inertial_measurement_unit
from easygopigo3 import *
import time

import picamera
import matplotlib.pyplot as plt
from PIL import Image
#import numpy as np
#from curtsies import Input
#from di_sensors import inertial_measurement_unit
from easygopigo3 import *


import csv


DEBUG = False # if set to True, any exception that's encountered is debugged
MAX_DISTANCE = 2300 # measured in mm
MIN_DISTANCE = 150 # measured in mm
NO_OBSTACLE = 3000
ERROR = 0 # the error that's returned when the DistanceSensor is not found
MAX_SPEED = 300 # max speed of the GoPiGo3
MIN_SPEED = 100 # min speed of the GoPiGo3
i = 0
# variable for triggering the closing procedure of the script
# used for stopping the while loop that's in the Main() function
robot_operating = True

        #TODO CODE: Define here so that initialized once and can be modifed later in square code
    
with open('project1_pathtrace.csv', 'w') as csvfile:
    fieldnames = ['row_num', 'encoder_value','distance_value','cardinal direction']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()


# handles the CTRL-C signal sent from the keyboard
# required for gracefull exits of the script
def signal_handler(signal, frame):
    global robot_operating
    print("CTRL-C combination pressed")
    robot_operating = False

# function for debugging
def debug(string):
    if DEBUG is True:
        print("Debug: " + str(string))
        
        
        
def Main():

    print("   _____       _____ _  _____         ____  ")
    print("  / ____|     |  __ (_)/ ____|       |___ \ ")
    print(" | |  __  ___ | |__) || |  __  ___     __) |")
    print(" | | |_ |/ _ \|  ___/ | | |_ |/ _ \   |__ < ")
    print(" | |__| | (_) | |   | | |__| | (_) |  ___) |")
    print("  \_____|\___/|_|   |_|\_____|\___/  |____/ ")
    print("                                            ")

    # initializing an EasyGoPiGo3 object and a DistanceSensor object
    # used for interfacing with the GoPiGo3 and with the distance sensor
    try:
        gopigo3 = EasyGoPiGo3()
        distance_sensor = gopigo3.init_distance_sensor()

    except IOError as msg:
        print("GoPiGo3 robot not detected or DistanceSensor not installed.")
        debug(msg)
        sys.exit(1)

    except FirmwareVersionError as msg:
        print("GoPiGo3 firmware needs to updated.")
        debug(msg)
        sys.exit(1)

    except Exception as msg:
        print("Error occurred. Set debug = True to see more.")
        debug(msg)
        sys.exit(1)

    if DEBUG is True:
        distance_sensor.enableDebug()

    # variable that says whether the GoPiGo3 moves or is stationary
    # used during the runtime
    gopigo3_stationary = True

    global robot_operating
    j = 1
    #4 cardinal NESW directions
    card_list = ['N','E','S','W']

    camPhotoTotal = 0
    time_to_wait_in_queue = 0.1 # 
    
    # while the script is running
    while robot_operating:
        # read the distance from the distance sensor
        dist = distance_sensor.read_mm()
        determined_speed = 0
        
        encoders_read = round(gopigo3.read_encoders_average())
            
            #write sensor values to a file
        global i
        row = [i, encoders_read, dist, j]
        with open('project1_pathtrace.csv', 'a') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)
                csvFile.close()
        i+=1

        
        
        if(encoders_read) <= 50:
                #if 50 cm crossed, stop and take right turn (NESW directions)
                gopigo3.stop()
                gopigo3.turn_degrees(90)

        # if the sensor can't be detected
        if dist == ERROR:
            print("Cannot reach DistanceSensor. Stopping the process.")
            robot_operating = False

        # if the robot is closer to the target
        if (dist <= 250):
            # then stop the GoPiGo3
            gopigo3_stationary = True
            #orbit around the obstacle at radius 30 deg 
            gopigo3_robot.orbit(180, 30)
            gopigo3.stop()

        # if the robot is far away from the target
        else:
            gopigo3_stationary = False

            # if the distance sensor can't detect any target
            if dist == NO_OBSTACLE:
                # then set the speed to maximum
                determined_speed = MAX_SPEED
            else:
                # otherwise, calculate the speed with respect to the distance from the target
                percent_speed = float(dist - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)
                determined_speed = MIN_SPEED + (MAX_SPEED - MIN_SPEED) * percent_speed

            # apply the changes
            gopigo3.set_speed(determined_speed)
            gopigo3.forward()

        # and last, print some stats
        
        #TODO CODE: Define here so that initialized once and can be modifed later in square code
    
    with open('project1_pathtrace.csv', 'w') as csvfile:
        fieldnames = ['row_num', 'encoder_value','distance_value','cardinal direction']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
    gopigo3.forward()
            current_distance = my_distance_sensor.read_mm()
            if current_distance < 500:
                pi_Photo(camPhotoTotal)
                camPhotoTotal = camPhotoTotal + 1
            print("Distance Sensor Reading: {} mm ".format(current_distance))
            sleep(0.001)
        if not simultaneous_launcher.broken:
            gopigo3.stop() 

            # command for making the robot move of stop
        if move is False:
            gopigo3.stop()
            
       
                                               

        print("Current distance : {:4} mm Current speed: {:4} Stopped: {}".format(current_distance, int(determined_speed), gopigo3_stationary is True ))

        # give it some time,
        # otherwise you'll have a hard time exiting the script
        sleep(0.08)

    # and finally stop the GoPiGo3 from moving
    gopigo3.stop()


def pi_Photo(camPhotoTotal):
    
    output = np.empty((480, 640, 3), dtype = np.uint8)
    with picamera.PiCamera() as camera:
        camera.resolution = (640, 480)
        camera.rotation = 180
        camera.resolution = (640,480)
    # for black and white uncomment the following
        camera.color_effects = (128,128)
        camera.capture(output, format = 'rgb', use_video_port = True)
    img = Image.fromarray(output)
    img.save("extracredit_problem2_{}.jpg".format(camPhotoTotal))    

if __name__ == "__main__":
    # signal handler
    # handles the CTRL-C combination of keys
    signal.signal(signal.SIGINT, signal_handler)
    Main()
