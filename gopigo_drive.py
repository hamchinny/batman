import time
from easygopigo3 import EasyGoPiGo3
from di_sensors.inertial_measurement_unit import InertialMeasurementUnit

class Robot:
    def __init__(self):
        self.gpg = EasyGoPiGo3()
        self.my_distance_sensor = self.gpg.init_distance_sensor('I2C')
        self.movement_history = []  # To track movements for rewinding

    def move_forward(self, distance):
        print(f"Moving forward {distance} inches.")
        self.gpg.drive_inches(distance)
        self.movement_history.append(("forward", distance))

    def turn(self, degrees):
        print(f"Turning {degrees} degrees.")
        self.gpg.turn_degrees(degrees)
        self.movement_history.append(("turn", degrees))

    def reverse_movements(self):
        print("Rewinding movements...")
        while self.movement_history:
            action, value = self.movement_history.pop()
            if action == "forward":
                print(f"Reversing {value} inches.")
                self.gpg.drive_inches(-value)
            elif action == "turn":
                print(f"Undoing turn of {value} degrees.")
                self.gpg.turn_degrees(-value)

    def execute_route(self):
        # Route logic from gopigo_drive.py
        self.turn(360)
        self.turn(360)
        self.turn(360)
        time.sleep(1)
        self.turn(180)
        self.move_forward(8)
        time.sleep(1)

        self.turn(360)
        self.turn(360)
        self.turn(360)
        time.sleep(1)
        self.move_forward(8)
        time.sleep(1)

        self.turn(360)
        self.turn(360)
        self.turn(360)
        time.sleep(1)
        self.turn(60)
        self.move_forward(8)
        time.sleep(1)

        self.turn(360)
        self.turn(360)
        self.turn(360)
        time.sleep(1)
        self.move_forward(8)
        time.sleep(1)

    def drive_and_monitor(self):
        try:
            while True:
                distance = self.my_distance_sensor.read_mm() / 10.0  # Convert mm to cm
                print(f"Distance: {distance} cm")
                if distance <= 15.0:  # Trigger threshold
                    print("Obstacle detected within 15 cm. Stopping and reversing.")
                    self.gpg.stop()
                    self.reverse_movements()  # Retrace movements
                    break  # Stop execution after reversing
                else:
                    print("Path clear. Following route.")
                    self.execute_route()
        except KeyboardInterrupt:
            print("Stopping robot due to user interruption.")
            self.gpg.stop()
        except Exception as e:
            print(f"An error occurred: {e}")
            self.gpg.stop()

if __name__ == "__main__":
    robot = Robot()
    robot.drive_and_monitor()
