
# Whiteboard Eraser Robot
![Whiteboard Eraser Robot](https://github.com/user-attachments/assets/3a8e6237-4c80-4d55-a5d9-cd29d9f72a1e)

---

## Overview
The **Whiteboard Eraser Robot** is an autonomous system designed to clean whiteboards efficiently using computer vision and LiDAR technology. The robot:
- Identifies written areas.
- Avoids clean sections.
- Navigates within defined boundaries.

---

## Project Requirements

### Software
- Python 3.x
- OpenCV
- NumPy
- RPLidar Python SDK
- imutils

### Hardware
- Servo motors (x2)
- LiDAR sensor
- Camera module
- Battery pack
- Microcontroller (e.g., Raspberry Pi)

---

## How to Use the Program

1. **Start the Robot**  
   Turn on the gopigo robot. Within 10–30 seconds, a green light will illuminate. After this, connect your computer to the gopigo Wi-Fi network.

2. **Verify Connection**  
   Once the gopigo successfully connects to your computer, a blue light will turn on.

3. **Access Control Interface**  
   Open your browser and go to [mygopigo.com](http://mygopigo.com) to control the robot.

4. **Activate Erasing Functionality**  
   - Connect the distance sensor to the robot.
   - Open the terminal and run the following command:
     ```bash
     python3 gopigo_drive.py
     ```

---


