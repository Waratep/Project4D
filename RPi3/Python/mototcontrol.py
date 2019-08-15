from asyncore import file_dispatcher, loop
from evdev import InputDevice, categorize, ecodes
import time
from adafruit_motorkit import MotorKit

dev = InputDevice('/dev/input/event0')
kit = MotorKit()


def map( x,  in_min,  in_max,  out_min,  out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


if __name__ == "__main__":

    x = y = z = 0
    mleft = mright = 0
    l2 = r2 = 0

    try:
        for event in dev.read_loop():

            kit.motor1.throttle = mleft
            kit.motor2.throttle = mright

            if event.type == ecodes.EV_KEY:
                btn = categorize(event).keycode
                if btn == 'BTN_TR':
                    r2 = event.value
                if btn == 'BTN_TL':
                    l2 = event.value

            if event.type == ecodes.EV_ABS:

                if event.code == 4 and r2 == 1: 
                    x = map(event.value,0,255,0,1.0)

                if event.code == 3 and l2 == 1: 
                    y = map(event.value,0,255,0,-1.0)

                if event.code == 0: 
                    z = map(event.value,0,255,-1.0,1.0)

            if z < 0:
                mleft = x + z 
                mright = x - z

            elif z > 0:
                mleft = x + z 
                mright = x - z
            else:
                mleft = x
                mright = x
            
            if mleft > 1.0:
                mleft = 1.0
            if mright > 1.0:
                mright = 1.0

            mleft = float("{0:.1f}".format(mleft))
            mright = float("{0:.1f}".format(mright))

            print(mleft,mright)

    except KeyboardInterrupt:
        print("ERROR")
        
