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

# import class imageprocessing and motor controller
from Imageprocessing import Imageprocessing
from MotorController import Motor


if __name__ == "__main__":
    
    motor = Motor()
    # imagepro = Imageprocessing()
    GPIO.add_event_detect(motor.left_encoder, GPIO.FALLING, motor.callbackleft_encoder, bouncetime=1)  
    GPIO.add_event_detect(motor.right_encoder, GPIO.FALLING, motor.callbackright_encoder, bouncetime=1)  

    try: 
        
        # imagepro._run(debugs = True)
        
        while (1): 
            
            motor.run()
            motor.setMotor(80,debug = True)

            time.sleep(0.1)

        motor.stop()

    except KeyboardInterrupt:
        motor.stop()
        print('KeyboardInterrupt')

    motor.stop()