######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/27/19
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
# This script will work with either a Picamera or regular USB webcam.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import random
from adafruit_motorkit import MotorKit
from pyzbar import pyzbar

def barcode(barcodes):
    bdata = None
    for barcode in barcodes:
        (x, y, w, h) = barcode.rect
        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type
        #text = "{} ({})".format(barcodeData, barcodeType)
        #cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
        #    0.5, (0, 0, 255), 2)
        #print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))
        bdata = barcodeData
        if bdata is not None:
            break
    
    # show the output image
    #cv2.imshow("Image", image)
    return bdata

def betweenOne(x):
    if x > 1:
        return 1
    if x < -1:
        return -1
    else:
        return x

def PID(offset,errorX,lasterror,kp,kd):
    error = float(errorX - offset)
    output = ( error * kp ) + ( ( error - lasterror ) * kd )
    return output

def redline(img):
    blue = [0,0,0]
    avg = 0
    for i in range(40):
        blue[0] += img[2][24+i][2]
        blue[1] += img[3][24+i][2]
        blue[2] += img[4][24+i][2]
        avg = (blue[0] + blue[1] + blue[2])/120
    return avg
    
def goForward():
    motor.motor2.throttle = 0.50
    motor.motor1.throttle = 0.50
    time.sleep(4)    
    
def goLeft():
    motor.motor2.throttle = 0.45
    motor.motor1.throttle = 0.45
    time.sleep(0.5)
    
    motor.motor2.throttle = 0.68
    motor.motor1.throttle = 0.35
    time.sleep(1.6)
    
def goRight():
    motor.motor2.throttle = 0.45
    motor.motor1.throttle = 0.45
    time.sleep(0.5)
    
    motor.motor2.throttle = 0.35
    motor.motor1.throttle = 0.60
    time.sleep(2.5)
    pass

def stop():
    motor.motor2.throttle = 0
    motor.motor1.throttle = 0

def process(frame):
    img = frame
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=4)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.8*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    edged = cv2.Canny(sure_fg, 30, 200)
    contours = cv2.findContours(edged,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    
    cv2.circle(edged,(contours[1][0][0][0][0],contours[1][0][0][0][1]), 5, (255,0,0), -1)
    cv2.circle(img,(contours[1][0][0][0][0],contours[1][0][0][0][1]), 5, (255,255,0), -1)

    return contours[1][0][0][0][0] , contours[1][0][0][0][1] ,  edged , sure_fg , img

def checkcrossroad(barcodeData):
    if barcodeData == '1':
        togo = random.randint(1, 2)
        print(barcodeData,togo)
        if togo == 1:
            goForward()
        elif togo == 2:
            goRight()  

    elif barcodeData == '2':
        togo = random.randint(1, 2)
        if togo == 1:
            goLeft()
        elif togo == 2:
            goRight()

    elif barcodeData == '3':
        togo = random.randint(1, 2)
        if togo == 1:
            goLeft()
        elif togo == 2:
            goForward()
    
    elif barcodeData == '4':
        togo = random.randint(1, 3)
        if togo == 1:
            goLeft()
        elif togo == 2:
            goForward()
        elif togo == 3:
            goRight()

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(100,70),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        #ret = self.stream.set(resolution[0])
        #ret = self.stream.set(resolution[1])
        #camera.resolution = (100, 70)
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(100,70),framerate=30).start()
time.sleep(1)

motor = MotorKit()
def_speed = 50
error = 0
last_error = 0
output = 0
kp = 1.2
kd = 1.5

running = True
goReadBar = False
res_change = False
turnBar = 0
timeforreadBar = 3000
millis = 0
startmillis = False
barcodeData = None

#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:

    # Grab frame from video stream
    frame1 = videostream.read()
    
    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (100, 70))
    input_data = np.expand_dims(frame_resized, axis=0)
    
    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

    # Draw framerate in corner of frame
    cv2.putText(frame_resized,'FPS: {0:.2f}'.format(frame_rate_calc),(10,10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),2,cv2.LINE_AA)

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame_resized)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break


    frame = frame_resized[40:70, 0:100]
    lasterror = error
    errorX , errorY , edged , sure_fg , img = process(frame)
    output = PID(50,errorX,lasterror,kp,kd)
    avg = redline(img)

    if int(avg) > 120:
        stop()
        running = False
        goReadBar = True
    else:
        motor.motor2.throttle = betweenOne((def_speed - output)/100)
        motor.motor1.throttle = betweenOne((def_speed + output)/100)
    
    if goReadBar:

        # cv2.destroyWindow("edged")
        # cv2.destroyWindow("sure_fg")
        # cv2.destroyWindow("img")
        # cv2.destroyWindow("original")

        if not res_change:
            camera.resolution = (640, 480)
            res_change = True
        
        if turnBar == 0:
            motor.motor2.throttle = 0.7
            motor.motor1.throttle = -0.7
            time.sleep(0.2)
            stop()
            turnBar = 1


        if turnBar == 1:
            if not startmillis:
                millis = int(round(time.time() * 1000))
                startmillis = True

            barcodes = pyzbar.decode(frame)
            if barcodeData is None:
                barcodeData = barcode(barcodes)
            else:
                barcodeData = barcodeData
                
            print('barcodeData',barcodeData)  

            if int(round(time.time() * 1000)) - millis > timeforreadBar:
                turnBar = 2
                

        if turnBar == 2:    
            motor.motor2.throttle = -0.7
            motor.motor1.throttle = 0.7
            time.sleep(0.2)
            stop()
            time.sleep(2)
            turnBar = 0
            goReadBar = False
            running = True
            startmillis = False
            if res_change:
                camera.resolution = (100, 70)
                res_change = False

    
    if barcodeData is not None and turnBar == 0:
        checkcrossroad(barcodeData)
        barcodeData = None

# Clean up
cv2.destroyAllWindows()
videostream.stop()
