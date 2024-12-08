import time
from math import atan2, pi
import subprocess
from easygopigo3 import EasyGoPiGo3
from di_sensors.inertial_measurement_unit import InertialMeasurementUnit

class Robot:
    def __init__(self):
        self.gpg = EasyGoPiGo3()
        self.imu = InertialMeasurementUnit(bus="GPG3_AD1")
        self.my_servo_portSERVO1 = self.gpg.init_servo('SERVO1')
        self.my_servo_portSERVO2 = self.gpg.init_servo('SERVO2')
        self.my_distance_sensor = self.gpg.init_distance_sensor('I2C')
        self.process = None  # To track the gopigo_drive.py process

    def monitor_distance_and_trigger(self):
        while True:
            distance = self.my_distance_sensor.read_mm() / 10.0  # Convert mm to cm
            print(f"Distance: {distance} cm")
            if distance <= 15.0:  # Trigger threshold
                print("Object detected within 15 cm. Restarting gopigo_drive.py.")
                # If a process is already running, terminate it
                if self.process is not None and self.process.poll() is None:
                    print("Stopping the current gopigo_drive.py process.")
                    self.process.terminate()
                    self.process.wait()  # Ensure the process has terminated
                # Start a new process
                self.process = subprocess.Popen(["python3", "./gopigo_drive.py"])
            time.sleep(0.5)  # Avoid rapid polling

if __name__ == "__main__":
    robot = Robot()
    robot.monitor_distance_and_trigger()
