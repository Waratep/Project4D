# import all of image processing
import cv2
import numpy as np
from scipy import ndimage as ndi
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from skimage.morphology import square,dilation
import copy
from scipy import ndimage as ndi

# import all of motor processing
from adafruit_motorkit import MotorKit
import RPi.GPIO as GPIO
import time

#import picamera
import picamera
import picamera.array

# import class imageprocessing and motor controller
from HoughTransform import HoughTransform
from MotorController import Motor


if __name__ == "__main__":
    
    motor = Motor()
    htf = HoughTransform()
    GPIO.add_event_detect(motor.left_encoder, GPIO.FALLING, motor.callbackleft_encoder, bouncetime=1)  
    GPIO.add_event_detect(motor.right_encoder, GPIO.FALLING, motor.callbackright_encoder, bouncetime=1)  

    try: 
        
        with picamera.PiCamera() as camera:
            with picamera.array.PiRGBArray(camera) as stream:
                camera.resolution = (100, 70)
                    
                while (1): 

                    camera.capture(stream, 'bgr', use_video_port=True)
                    image = stream.array
                    # cv2.imshow('frame',image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    htf.run(image,debugs = True)
                    # reset the stream before the next capture
                    stream.seek(0)
                    stream.truncate()

                cv2.destroyAllWindows()

        motor.stop()

    except KeyboardInterrupt:
        motor.stop()
        print('KeyboardInterrupt')

    motor.stop()