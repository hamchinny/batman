
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
import numpy as np
import cv2 as cv
import IPython
import io
import time
from PIL import Image
import time
import threading
import queue
from sklearn.cluster import KMeans


MINIMUM_VOLTAGE = 7.0
DEBUG = False
MOTORS_SPEED = 250  # see documentation
# TODO###############
import csv

# index number for all loops
i = 0
# OPTIONAL###########
# for Georgia  -5° 29'
MAGNETIC_DECLINATION = -5
# TODO###############


def getNorthPoint(imu):
    """
    Determines the heading of the north point.
    This function doesn't take into account the declination.

    :param imu: It's an InertialMeasurementUnit object.
    :return: The heading of the north point measured in degrees. The north point is found at 0 degrees.

    """

    x, y, z = imu.read_magnetometer()

    # using the x and z axis because the sensor is mounted vertically
    # the sensor's top face is oriented towards the back of the robot
    heading = -atan2(x, -z) * 180 / pi

    # adjust it to 360 degrees range
    if heading < 0:
        heading += 360
    elif heading > 360:
        heading -= 360

    # when the heading is towards the west the heading is negative
    # when the heading is towards the east the heading is positive
    if 180 < heading <= 360:
        heading -= 360

    heading += MAGNETIC_DECLINATION

    return heading


def statisticalNoiseReduction(values, std_factor_threshold=2):
    """
    Eliminates outlier values that go beyond a certain threshold.

    :param values: The list of elements that are being filtered.
    :param std_factor_threshold: Filtering aggressiveness. The bigger the value, the more it filters.
    :return: The filtered list.

    """

    if len(values) == 0:
        return []

    valarray = np.array(values)
    mean = valarray.mean()
    standard_deviation = valarray.std()
    # just return if we only got constant values
    if standard_deviation == 0:
        return values

    # remove outlier values
    valarray = valarray[
        (valarray > mean - std_factor_threshold * standard_deviation)
        & (valarray < mean + std_factor_threshold * standard_deviation)
    ]

    return list(valarray)


# TODOCODE##########################################

# Use 'jpeg' instead of 'png' (~5 times faster)
def showarray(a, fmt="jpeg"):
    """
    Function to display an image within a Jupyter notebook.
    """
    f = io.BytesIO()
    Image.fromarray(a).save(f, fmt)
    IPython.display.display(
        IPython.display.Image(data=f.getvalue(), width=480, height=360)
    )


def resizeNPArray(array, width, height):
    """
    Function to resize a given numpy array to another width/height,
    whilst preserving the relative information - used for images.
    """
    img = Image.fromarray(array)
    img = img.resize((width, height), Image.ANTIALIAS)
    resized = np.asarray(img)
    return resized


def detectFacesAndEyes(img_array):
    """
    Function to detect eyes and faces using a Haar-Cascade classifier.
    """
    gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv.rectangle(img_array, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = img_array[y : y + h, x : x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)


def squarepath(trigger):
    gopigo3_robot = EasyGoPiGo3()
    my_distance_sensor = gopigo3_robot.init_distance_sensor()

    j = 1
    # 4 cardinal NESW directions
    card_list = ["N", "E", "S", "W"]

    def drive_and_turn(j, dist=999):
        gopigo3_robot.reset_encoders()
        # drive autonomously until Ctrl-C pressed
        # check if an obstacle faced within 250mm (25cm)
        while not (trigger.is_set() or dist <= 250):
            gopigo3_robot.forward()
            dist = my_distance_sensor.read_mm()
            print("Distance Sensor Reading: {} mm ".format(dist))
            # enocder average values to get distance in cm moved
            encoders_read = round(gopigo3_robot.read_encoders_average())

            # write sensor values to a file
            global i
            row = [i, encoders_read, dist, j]
            with open("probl2_pathtrace.csv", "a") as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)
                csvFile.close()
            i += 1

            if (encoders_read) >= 50:
                # if 50 cm crossed, stop and take right turn (NESW directions)
                gopigo3_robot.stop()
                gopigo3_robot.turn_degrees(90)

                break
        if dist < 250:
            # when object within 25cm encountered
            gopigo3_robot.stop()
            # orbit around the obstacle at radius 30 deg
            gopigo3_robot.orbit(180, 30)
            gopigo3_robot.stop()
            return
        else:
            return

    for x in range(4):
        drive_and_turn(card_list[x])

    # return to main func from where we start square from North again
    return


# TODOCODE################################################################333


def orientate(trigger, simultaneous_launcher, sensor_queue):
    """
    Thread-launched function for reading the compass data off of the IMU sensor. The data is then
    interpreted and then it's loaded in a queue.

    :param trigger: CTRL-C event. When it's set, it means CTRL-C was pressed and the thread needs to stop.
    :param simultaneous_launcher: It's a barrier used for synchronizing all threads together.
    :param sensor_queue: Queue where the processed data of the compass is put in.
    :return: Nothing.

    """

    time_to_put_in_queue = 0.2  # measured in seconds
    time_to_wait_after_error = 0.5  # measured in seconds

    # try instantiating an InertialMeasurementUnit object
    try:
        imu = inertial_measurement_unit.InertialMeasurementUnit(bus="GPG3_AD1")
    except Exception as msg:
        print(str(msg))
        simultaneous_launcher.abort()

    # start the calibrating process of the compass
    print("Rotate the GoPiGo3 robot with your hand until it's fully calibrated")
    try:
        compass = imu.BNO055.get_calibration_status()[3]
    except Exception as msg:
        compass = 0
    values_already_printed = []
    max_conseq_errors = 3

    while compass != 3 and not trigger.is_set() and max_conseq_errors > 0:
        state = ""
        if compass == 0:
            state = "not yet calibrated"
        elif compass == 1:
            state = "partially calibrated"
        elif compass == 2:
            state = "almost calibrated"

        if not compass in values_already_printed:
            print("The GoPiGo3 is " + state)
        values_already_printed.append(compass)

        try:
            compass = imu.BNO055.get_calibration_status()[3]
        except Exception as msg:
            max_conseq_errors -= 1
            sleep(time_to_wait_after_error)
            continue

    # if CTRL-C was triggered or if the calibration failed
    # then abort everything
    if trigger.is_set() or max_conseq_errors == 0:
        print("IMU sensor is not reacheable or kill event was triggered")
        simultaneous_launcher.abort()
    else:
        state = "fully calibrated"
        print("The GoPiGo3 is " + state)

    # point of synchronizing all threads together (including main)
    # it fails if abort method was called
    try:
        simultaneous_launcher.wait()
    except threading.BrokenBarrierError as msg:
        print("[orientate] thread couldn't fully start up")

    # while CTRl-C is not pressed and while the synchronization went fine
    while not (trigger.is_set() or simultaneous_launcher.broken):
        five_values = 10
        heading_list = []
        max_conseq_errors = 3

        # get the north point
        # extract a couple of values before going to the next procedure
        while five_values > 0 and max_conseq_errors > 0:
            try:
                heading_list.append(getNorthPoint(imu))
            except Exception as msg:
                max_conseq_errors -= 1
                sleep(time_to_wait_after_error)
                continue
            five_values -= 1
        if max_conseq_errors == 0:
            print("IMU is not reacheable")
            trigger.set()
            break

        # apply some filtering
        heading_list = statisticalNoiseReduction(heading_list)
        heading_avg = mean(heading_list)

        # and then try to put it in the queue
        # if the queue is full, then just go to the next iteration of the while loop
        try:
            sensor_queue.put(heading_avg, timeout=time_to_put_in_queue)
        except queue.Full:
            pass


def robotControl(trigger, simultaneous_launcher, motor_command_queue, sensor_queue):
    """
    Thread-launched function for orientating the robot around. It gets commands from the keyboard as well
    as data from the sensor through the sensor_queue queue.

    :param trigger: CTRL-C event. When it's set, it means CTRL-C was pressed and the thread needs to stop.
    :param simultaneous_launcher: It's a barrier used for synchronizing all threads together.
    :param motor_command_queue: Queue containing commands from the keyboard. The commands are read from within main.
    :param sensor_queue: Processed data off of the IMU. The queue is intended to be read.
    :return: Nothing.

    """

    time_to_wait_in_queue = 0.1  # measured in

    # try to connect to the GoPiGo3
    try:
        gopigo3_robot = EasyGoPiGo3()
        # TODO CODE################
        # my_distance_sensor = gopigo3_robot.init_distance_sensor()
        # TODO CODE################
    except IOError:
        print("GoPiGo3 robot not detected")
        simultaneous_launcher.abort()
    except gopigo3.FirmwareVersionError:
        print("GoPiGo3 board needs to be updated")
        simultaneous_launcher.abort()
    except Exception:
        print("Unknown error occurred while instantiating GoPiGo3")
        simultaneous_launcher.abort()

    # synchronizing point between all threads
    # if abort method was called, then the synch will fail
    try:
        simultaneous_launcher.wait()
    except threading.BrokenBarrierError as msg:
        print("[robotControl] thread couldn't be launched")

    # if threads were successfully synchronized
    # then set the GoPiGo3 appropriately
    if not simultaneous_launcher.broken:
        gopigo3_robot.stop()
        gopigo3_robot.set_speed(MOTORS_SPEED)

    direction_degrees = None
    move = False
    acceptable_error_percent = 8
    command = "stop"
    rotational_factor = 0.30
    accepted_minimum_by_drivers = 6

    # TODO CODE: Define here so that initialized once and can be modifed later in square code

    with open("problem2_pathtrace.csv", "w") as csvfile:
        fieldnames = [
            "row_num",
            "encoder_value",
            "distance_value",
            "cardinal direction",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    # while CTRL-C is not pressed, the synchronization between threads didn't fail and while the batteries' voltage isn't too low
    while not (
        trigger.is_set()
        or simultaneous_launcher.broken
        or gopigo3_robot.volt() <= MINIMUM_VOLTAGE
    ):
        # read from the queue of the keyboard
        try:
            command = motor_command_queue.get(timeout=time_to_wait_in_queue)
            motor_command_queue.task_done()
        except queue.Empty:
            pass

        # make some selection depending on what every command represents
        if command == "stop":
            move = False
        elif command == "move":
            move = True
        if command == "west":
            direction_degrees = -90.0
        elif command == "east":
            direction_degrees = 90.0
        elif command == "north":
            # TODOCODE#############
            # let us start square traversal once set as North
            move = True
            direction_degrees = 0.0
        elif command == "south":
            direction_degrees = 180.0

        # if a valid orientation was selected
        if direction_degrees is not None:
            # read data and calculate orientation
            heading = sensor_queue.get()
            if direction_degrees == 180.0:
                heading_diff = (direction_degrees - abs(heading)) * (
                    -1 if heading < 0 else 1
                )
                error = abs(heading_diff / direction_degrees) * 100
            else:
                heading_diff = direction_degrees - heading
                error = abs(heading_diff / 180) * 100

            how_much_to_rotate = int(heading_diff * rotational_factor)

            if DEBUG is True:
                print(
                    "direction_degrees {} heading {} error {} heading_diff {}".format(
                        direction_degrees, heading, error, heading_diff
                    )
                )

            # check if the heading isn't so far from the desired orientation
            # if it needs correction, then rotate the robot
            if (
                error >= acceptable_error_percent
                and abs(how_much_to_rotate) >= accepted_minimum_by_drivers
            ):
                gopigo3_robot.turn_degrees(how_much_to_rotate, blocking=True)

        # command for making the robot move of stop
        if move is False:
            gopigo3_robot.stop()
        else:
            # TODOCODE###########
            # starts once NORTH is set
            squarepath(trigger)
            # TODOCODE###########

        sleep(0.001)

    # if the synchronization wasn't broken
    # then stop the motors in case they were running
    if not simultaneous_launcher.broken:
        gopigo3_robot.stop()


class ImageProcessor(threading.Thread, trigger):
    """
    Thread-safe class to process a stream of jpeg sequences from a given queue.
    """

    def __init__(self, thread_stopper, frames, lock):
        """
        thread_stopper -> Is the event which stops the thread when set.
        frames -> The queue from which jpeg images come (in numpy.array format).
        lock -> Mutex for the queue.
        """
        super().__init__()
        self.thread_stopper = thread_stopper
        self.frames = frames
        self.lock = lock
        self.incoming = np.empty((240, 320, 3), dtype=np.uint8)
        self.processed = np.zeros((240, 320, 3), dtype=np.uint8)

        self.verticals = np.array(80 * [np.arange(0, 60)]).T
        self.verticals = self.verticals[:, :, np.newaxis]

        self.horizontals = np.array(60 * [np.arange(0, 80)])
        self.horizontals = self.horizontals[:, :, np.newaxis]

    def run(self):
        """
        Main thread which runs indefinitely until <<thread_stopper>> event is set.
        This function processes each incoming image from the queue iteratively and then displays it in this notebook.
        """
        while not thread_stopper.is_set():
            try:
                self.lock.acquire()
                self.incoming = self.frames.get_nowait()
                self.position, self.processed = self.dowork(self.incoming)
                self.frames.task_done()
            except queue.Empty:
                continue
            finally:
                self.lock.release()
            showarray(self.processed)
            IPython.display.clear_output(wait=True)

    def dowork(self, array):
        """
        array -> Is a numpy array that holds the a RGB image.
        Function to process an image and detect spots of a given targeted color.
        """

        # down-sizing the image and running KMeans on it
        output = array.copy()
        array = resizeNPArray(array, 80, 60)
        image_and_positions = np.concatenate(
            (array, self.verticals, self.horizontals), axis=2
        )
        reshaped = image_and_positions.reshape((60 * 80, 5))
        kmeans = KMeans(
            n_clusters=6, n_init=1, max_iter=300, precompute_distances=True
        ).fit(reshaped)
        rgb_centers = kmeans.cluster_centers_[:, 0:3]

        labels_rgb = np.empty((4800, 3))
        for i in range(6):
            labels_rgb[kmeans.labels_ == i] = rgb_centers[i]
        labels_rgb = labels_rgb.reshape((60, 80, 3)).astype(np.uint8)

        # getting the closest KMeans center to the targeted color
        diff = rgb_centers - target_color
        closest = np.sqrt(np.power(diff, 2).sum(axis=1))
        closest_label = closest.argmin()

        # determining the distribution of the targeted pixels
        # (the target pixels are identified with the label of the selected KMeans center)
        labels = kmeans.labels_.reshape((60, 80))
        labels = labels == closest_label
        sum_labels_vertical = labels.sum(axis=1)
        sum_labels_horizontal = labels.sum(axis=0)

        # 4800 = 60 * 80 pixels
        if not sum_labels_vertical.sum() > color_threshold * 4800:
            return (None, output)

        # find the countour of the spot of color
        non_zero_elements = np.nonzero(sum_labels_vertical)
        # multiply by 4 to get to the original size
        min_vertical = np.min(non_zero_elements) * 4
        max_vertical = np.max(non_zero_elements) * 4
        non_zero_elements = np.nonzero(sum_labels_horizontal)
        min_horizontal = np.min(non_zero_elements) * 4
        max_horizontal = np.max(non_zero_elements) * 4

        # and then draw a rectangle around the detected spot of color
        output[min_vertical : max_vertical + 1, min_horizontal, :] = border_color
        output[min_vertical : max_vertical + 1, max_horizontal, :] = border_color
        output[min_vertical, min_horizontal : max_horizontal + 1, :] = border_color
        output[max_vertical, min_horizontal : max_horizontal + 1, :] = border_color

        center_position = (min_vertical + max_vertical) / 2

        return (center_position, output)


face_cascade = cv.CascadeClassifier("haarcascade_frontalface_alt.xml")
eye_cascade = cv.CascadeClassifier("haarcascade_eye.xml")
frames = queue.Queue(maxsize=10)
thread_stopper = threading.Event()
lock = threading.Lock()
target_color = np.array([0, 0, 255])
# target_color = np.array([255, 255, 255]) # white
border_color = np.array([0, 255, 0])  # green
color_threshold = 0.07  # in percentage
time_to_run = 50  # in seconds

start = time.time()
imageThread = ImageProcessor(thread_stopper, frames, lock)
imageThread.start()

with picamera.PiCamera() as camera:
    camera.resolution = (320, 240)
    camera.framerate = 30
    while time.time() - start < time_to_run:
        freshest_frame = np.empty((240, 320, 3), dtype=np.uint8)
        camera.capture_sequence([freshest_frame], use_video_port=True, format="rgb")
        detectFacesAndEyes(freshest_frame)
        showarray(freshest_frame)
        IPython.display.clear_output(wait=True)
        lock.acquire()
        if frames.full():
            frames.get()
            frames.task_done()
        else:
            frames.put(freshest_frame)
        lock.release()
    print("picamera session ended")

    thread_stopper.set()
    print("triggered image processing thread")

    imageThread.join()
    print("thread joined")

    simultaneous_launcher = threading.Barrier(3)  # synchronization object
    motor_command_queue = queue.Queue(maxsize=2)  # queue for the keyboard commands
    sensor_queue = queue.Queue(maxsize=1)  # queue for the IMU sensor
    keyboard_refresh_rate = 20.0  # how many times a second the keyboard should update
    available_commands = {
        "<LEFT>": "west",
        "<RIGHT>": "east",
        "<UP>": "north",
        "<DOWN>": "south",
        "<SPACE>": "stop",
        "w": "move",
    }  # the selectable options within the menu
    menu_order = [
        "<LEFT>",
        "<RIGHT>",
        "<UP>",
        "<DOWN>",
        "<SPACE>",
        "w",
    ]  # and the order of these options

    print("   _____       _____ _  _____         ____  ")
    print("  / ____|     |  __ (_)/ ____|       |___ \ ")
    print(" | |  __  ___ | |__) || |  __  ___     __) |")
    print(" | | |_ |/ _ \|  ___/ | | |_ |/ _ \   |__ < ")
    print(" | |__| | (_) | |   | | |__| | (_) |  ___) |")
    print("  \_____|\___/|_|   |_|\_____|\___/  |____/ ")
    print("                                            ")

    # starting the workers/threads
    orientate_thread = threading.Thread(
        target=orientate, args=(trigger, simultaneous_launcher, sensor_queue)
    )
    robotcontrol_thread = threading.Thread(
        target=robotControl,
        args=(trigger, simultaneous_launcher, motor_command_queue, sensor_queue),
    )
    orientate_thread.start()
    robotcontrol_thread.start()

    # if the threads couldn't be launched, then don't display anything else
    try:
        simultaneous_launcher.wait()

        print(
            "Press the following keys for moving/orientating the robot by the 4 cardinal points"
        )
        for menu_command in menu_order:
            print("{:8} - {}".format(menu_command, available_commands[menu_command]))
    except threading.BrokenBarrierError:
        pass

    # read the keyboard as long as the synchronization between threads wasn't broken
    # and while CTRL-C wasn't pressed
    with Input(keynames="curtsies") as input_generator:
        while not (trigger.is_set() or simultaneous_launcher.broken):
            period = 1 / keyboard_refresh_rate
            key = input_generator.send(period)

            if key in available_commands:
                try:
                    motor_command_queue.put_nowait(available_commands[key])
                except queue.Full:
                    pass

    # exit codes depending on the issue
    if simultaneous_launcher.broken:
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    trigger = threading.Event()  # event used when CTRL-C is pressed
    signal.signal(
        signal.SIGINT, lambda signum, frame: trigger.set()
    )  # SIGINT (CTRL-C) signal handler
Main(trigger)
