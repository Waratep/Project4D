from asyncore import file_dispatcher, loop
from evdev import InputDevice, categorize, ecodes
import time
from adafruit_motorkit import MotorKit

import picamera     
from time import sleep  

dev = InputDevice('/dev/input/event0')
kit = MotorKit()


def map( x,  in_min,  in_max,  out_min,  out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

if __name__ == "__main__":

    x = y = z = 0
    mleft = mright = 0
    l2 = r2 = 0

    camera = picamera.PiCamera()   
    camera.close()
    camera = picamera.PiCamera()  
    
    rec = 0

    start = time.time()
    try:
        for event in dev.read_loop():
            
            if time.time() - start >= 25 and rec == 1:
                camera.stop_recording()
                rec = 2

            kit.motor1.throttle = mleft
            kit.motor2.throttle = mright
            
            if event.type == ecodes.EV_KEY:
                btn = categorize(event).keycode
                if btn == 'BTN_TR':
                    r2 = event.value
                if btn == 'BTN_TL':
                    l2 = event.value

                if btn[0] == 'BTN_B':
                    # camera.start_preview()      
                    if rec == 0:
                        camera.start_recording('/home/pi/Desktop/video1.mjpeg') 
                        rec = 1
                    print("X")
                if btn == 'BTN_C':
                    camera.stop_recording()
                    # camera.stop_preview()
                    print("O")

            if event.type == ecodes.EV_ABS:

                if event.code == 4 and r2 == 1: 
                    x = map(event.value,0,255,0,0.5)

                if event.code == 3 and l2 == 1: 
                    y = map(event.value,0,255,0,-0.5)

                if event.code == 0: 
                    z = map(event.value,0,255,-0.5,0.5)

            if z < 0:
                mleft = x + z 
                mright = x - z

            elif z > 0:
                mleft = x + z 
                mright = x - z
            else:
                mleft = x
                mright = x
            
            if mleft > 0.5:
                mleft = 0.5
            if mright > 0.5:
                mright = 0.5

            mleft = float("{0:.1f}".format(mleft))
            mright = float("{0:.1f}".format(mright)) + 0.05

            print(mleft,mright)

    except KeyboardInterrupt:
        camera.close()
        print("ERROR")
        
