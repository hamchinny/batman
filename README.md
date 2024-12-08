# Whiteboard Eraser Robot
![image](https://github.com/user-attachments/assets/3a8e6237-4c80-4d55-a5d9-cd29d9f72a1e)


Overview
This project involves designing an Autonomous Whiteboard Erasing Robot that leverages computer vision and LiDAR technology to clean whiteboards efficiently. The robot identifies written areas, avoids clean sections, and navigates within defined boundaries.

Project Requirements
Software
Python 3.x
OpenCV
NumPy
RPLidar Python SDK
imutils
Hardware
Servo motors (x2)
LiDAR sensor
Camera module
Battery pack
Microcontroller (e.g., Raspberry Pi)
Sandbox edge enclosure

## How to Use the Program
1. When the gopigo robot is started, a green light will turn on within 10–30 seconds. After that, connect the computer to the gopigo Wi-Fi.
2. Once the gopigo successfully connects to the computer, a blue light will illuminate on the gopigo robot.
3. Then, go to [mygopigo.com](http://mygopigo.com) to control the robot.
4. After connecting the distance sensor, run `python3 raider.py` in the terminal. This will execute the `raider.py` script. When the sensor detects an object, `gopigo_drive.py` will execute, allowing the robot to move and erase marks on the whiteboard.
