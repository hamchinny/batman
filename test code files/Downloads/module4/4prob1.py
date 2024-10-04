from easygopigo3 import EasyGoPiGo3
from gopigo3 import FirmwareVersionError
import sys
import signal
from time import sleep
import time
import threading
#import Motor_Turn.py
import picamera
import numpy as np
import cv2 as cv
import IPython
import io
from PIL import Image

DEBUG = False # if set to True, any exception that's encountered is debugged
MAX_DISTANCE = 2300 # measured in mm
MIN_DISTANCE = 100 # measured in mm
NO_OBSTACLE = 3000
ERROR = 0 # the error that's returned when the DistanceSensor is not found
MAX_SPEED = 300 # max speed of the GoPiGo3
MIN_SPEED = 100 # min speed of the GoPiGo3



# variable for triggering the closing procedure of the script
# used for stopping the while loop that's in the Main() function
robot_operating = True

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
        
# function for counting 

def showarray(a, fmt='jpeg', ):
    '''
    Function to display an image within a Jupyter notebook.
    '''
    f = io.BytesIO()
    f2 = io.BytesIO()
    Image.fromarray(a).save(f, fmt)
    IPython.display.display(IPython.display.Image(data=f.getvalue(), width = 480, height = 360))
    Image.fromarray(a).save(f2, fmt)
    IPython.display.display(IPython.display.Image(data=f2.getvalue(), width = 480, height = 360))
    
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
    eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

def resizeNPArray(array, width, height):
    '''
    Function to resize a given numpy array to another width/height,
    whilst preserving the relative information - used for images.
    '''
    img = Image.fromarray(array)
    img = img.resize((width, height), Image.ANTIALIAS)
    resized = np.asarray(img)
    return resized
    
def detectFacesAndEyes(img_array):
    '''
    Function to detect eyes and faces using a Haar-Cascade classifier.
    '''
    gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv.rectangle(img_array,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img_array[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
               
            
    with picamera.PiCamera() as camera:
        camera.resolution = (320, 240)
        camera.framerate = 30
        freshest_frame = np.empty((240, 320, 3), dtype = np.uint8)
        while True:
            camera.capture(freshest_frame, use_video_port = True, format = 'rgb')
            detectFacesAndEyes(freshest_frame)
            showarray(freshest_frame)
            IPython.display.clear_output(wait = True)
    img = Image.fromarray(output)
    
    img.save("Module4_prob1_{}.jpg".format(camPhotoTotal))


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
        
    
    
                   
            #for i in range(-360, 361):
                #gopigo3.set_motor_position(gopigo3.MOTOR_LEFT + gopigo3.MOTOR_RIGHT, -i)
                #time.sleep(0.01)
        
            #for i in range(-360, 361):
                #gopigo3.set_motor_position(gopigo3.MOTOR_LEFT + gopigo3.MOTOR_RIGHT, i)
                #time.sleep(0.01)
    

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

    # while the script is running
    while robot_operating:
        # read the distance from the distance sensor
        current_distance = distance_sensor.read_mm()
        determined_speed = 0
        gopigo3.reset_encoders()
        totalDist = 0
        countTurn = 0
    
          
           
        # if the sensor can't be detected
        if current_distance == ERROR:
            print("Cannot reach DistanceSensor. Stopping the process.")
            robot_operating = False
        
       
        # if the robot is closer to the target
        elif current_distance < MIN_DISTANCE: #add stuff here 
            # then stop the GoPiGo3
            gopigo3_stationary = True
            gopigo3.turn_degrees(-200)
            gopigo3.stop()
           
        
           
        
        # if the robot is far away from the target
        else:
            gopigo3_stationary = False        
           
                    
            # if the distance sensor can't detect any target
            if current_distance == NO_OBSTACLE:
                # then set the speed to maximum
                determined_speed = MAX_SPEED
                 #while running 
               
            else:
                # otherwise, calculate the speed with respect to the distance from the target
                percent_speed = float(current_distance - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)
                determined_speed = MIN_SPEED + (MAX_SPEED - MIN_SPEED) * percent_speed

            # apply the changes 
            gopigo3.set_speed(determined_speed)
            #encoders_read = round(gopigo3.read_encoders_average())
        
            gopigo3.forward()
            #gopigo3.drive_cm(50)
            encoders_read = round(gopigo3.read_encoders_average())
            #gopigo3.stop()
            #gopigo3.turn_degrees(90)
            #gopigo3.drive_cm(30)
            #gopigo3.stop()
            #gopigo3.turn_degrees(-90)
            #gopigo3.drive_cm(50)
            #gopigo3.stop()
            #gopigo3.turn_degrees(-90)
            #gopigo3.drive_cm(15)
            #gopigo3.stop()
            #gopigo3.turn_degrees(90)
           
            #gopigo3.stop()
            #maybe stop 
        
        data = ("Current distance : {:4} mm Current speed: {:4} Stopped: {}".format(current_distance, int(determined_speed), gopigo3_stationary is True ))
        
        encoderV = ("Drove {:.2f} cm".format(encoders_read))
        print(data)
        print(encoderV)
        
        ##index = 1
        #for i in data:
           # i = index
            #i+=1    
            #print( index + data)
           
        file = open("problem2_pathtrace.csv", "a+")
        file.write(data)
        file.write(encoderV)
        
        #for i in range(1,{}):
        
         
        # and last, print some stats
   
        # give it some time,
        # otherwise you'll have a hard time exiting the script
        sleep(.05)

    # and finally stop the GoPiGo3 from moving
    gopigo3.stop()


if __name__ == "__main__":
    # signal handler
    # handles the CTRL-C combination of keys
    signal.signal(signal.SIGINT, signal_handler)
    Main()

    
class ImageProcessor(threading.Thread):
    '''
    Thread-safe class to process a stream of jpeg sequences from a given queue.
    '''
    def __init__(self, thread_stopper, frames, lock):
        '''
        thread_stopper -> Is the event which stops the thread when set.
        frames -> The queue from which jpeg images come (in numpy.array format).
        lock -> Mutex for the queue.
        '''
        super().__init__()
        self.thread_stopper = thread_stopper
        self.frames = frames
        self.lock = lock
        self.incoming = np.empty((240, 320, 3), dtype = np.uint8)
        self.processed = np.zeros((240, 320, 3), dtype = np.uint8)
        
        self.verticals = np.array(80 * [np.arange(0, 60)]).T
        self.verticals = self.verticals[:,:,np.newaxis]
        
        self.horizontals = np.array(60 * [np.arange(0, 80)])
        self.horizontals = self.horizontals[:,:,np.newaxis]
        
    def run(self):
        '''
        Main thread which runs indefinitely until <<thread_stopper>> event is set.
        This function processes each incoming image from the queue iteratively and then displays it in this notebook.
        '''
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
            IPython.display.clear_output(wait = True)
            
    def dowork(self, array):
        '''
        array -> Is a numpy array that holds the a RGB image.
        Function to process an image and detect spots of a given targeted color.
        '''
        
        # down-sizing the image and running KMeans on it
        output = array.copy()
        array = resizeNPArray(array, 80, 60)
        image_and_positions = np.concatenate((array, self.verticals, self.horizontals), axis = 2)
        reshaped = image_and_positions.reshape((60 * 80, 5))
        kmeans = KMeans(n_clusters = 6,
                       n_init = 1,
                       max_iter = 300,
                       precompute_distances = True).fit(reshaped)
        rgb_centers = kmeans.cluster_centers_[:, 0:3]
        
        labels_rgb = np.empty((4800, 3))
        for i in range(6):
            labels_rgb[kmeans.labels_ == i] = rgb_centers[i]
        labels_rgb = labels_rgb.reshape((60, 80, 3)).astype(np.uint8)
        
        # getting the closest KMeans center to the targeted color
        diff = rgb_centers - target_color
        closest = np.sqrt(np.power(diff, 2).sum(axis = 1))
        closest_label = closest.argmin()
        
        # determining the distribution of the targeted pixels
        # (the target pixels are identified with the label of the selected KMeans center)
        labels = kmeans.labels_.reshape((60, 80))
        labels = labels == closest_label
        sum_labels_vertical = labels.sum(axis = 1)
        sum_labels_horizontal = labels.sum(axis = 0)
        
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
        output[min_vertical:max_vertical+1,min_horizontal,:] = border_color
        output[min_vertical:max_vertical+1,max_horizontal,:] = border_color
        output[min_vertical,min_horizontal:max_horizontal+1,:] = border_color
        output[max_vertical,min_horizontal:max_horizontal+1,:] = border_color
        
        center_position = (min_vertical + max_vertical) / 2
                
        return (center_position, output)
    
frames = queue.Queue(maxsize = 10)
thread_stopper = threading.Event()
lock = threading.Lock()
target_color = np.array([255, 255, 255]) # white
border_color = np.array([0, 255, 0]) # green
color_threshold = 0.07 # in percentage
time_to_run = 30 # in seconds

start = time.time()
imageThread = ImageProcessor(thread_stopper, frames, lock)
imageThread.start()

with picamera.PiCamera() as camera:
    camera.resolution = (320, 240)
    camera.framerate = 30
    while time.time() - start < time_to_run:
        freshest_frame = np.empty((240, 320, 3), dtype = np.uint8)
        camera.capture_sequence([freshest_frame], use_video_port = True, format = 'rgb')
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