import numpy as np
import cv2
import time
import picamera
import picamera.array
from adafruit_motorkit import MotorKit
from pyzbar import pyzbar
import argparse
import random

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

if __name__ == "__main__":

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

    try: 
        
        camera = picamera.PiCamera()
        stream = picamera.array.PiRGBArray(camera)
        camera.resolution = (100, 70)
        while 1:

            camera.capture(stream, 'rgb', use_video_port=True)
            frame = stream.array
            cv2.imshow('frame', frame)

            if running:
                
                original = frame
                frame = frame[40:70, 0:100]
                lasterror = error
                errorX , errorY , edged , sure_fg , img = process(frame)
                # cv2.imshow("edged", edged)
                # cv2.imshow("sure_fg", sure_fg)
                # cv2.imshow('img', img)
                # cv2.imshow('original', original)
                output = PID(50,errorX,lasterror,kp,kd)
                avg = redline(img)
        
                if int(avg) > 120:
                    stop()
                    running = False
                    goReadBar = True
                    #readbar(image)
                    #goForward()
                    #goRight()
                    #goLeft()
                    
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

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            stream.seek(0)
            stream.truncate()

        stop()
        cv2.destroyAllWindows()

    except KeyboardInterrupt:
        stop()
        print('KeyboardInterrupt')
