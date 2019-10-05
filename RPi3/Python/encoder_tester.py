from adafruit_motorkit import MotorKit
import RPi.GPIO as GPIO
import threading
import time
kit = MotorKit()

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM) 
GPIO.setup(21, GPIO.IN)


def encoder():
    millis = int(round(time.time() * 1000))
    counter = 0
    while 1:
        if (GPIO.input(21)):
            counter += 1
        if (int(round(time.time() * 1000)) - millis >= 1000):
            millis = int(round(time.time() * 1000))
            print(counter)
            counter = 0
        

def control():
    while 1:
        i = input()
        # i = input()
        if i == 'w':
            # kit.motor1.throttle = 1
            kit.motor2.throttle = 1
        if i == 's':
            # kit.motor1.throttle = 0
            kit.motor2.throttle = 0


en = threading.Thread(target=encoder)
cn = threading.Thread(target=control)

if __name__ == "__main__":

    counter = 0
    state = 0

    level = 1.0
    timer = 0
    sum = 0
    try:

        kit.motor1.throttle = level
        millis = int(round(time.time() * 1000))

        while (timer < 60):
            print(timer)
            if (int(round(time.time() * 1000)) - millis >= 1000):
                millis = int(round(time.time() * 1000))
                sum += counter
                counter = 0
                timer += 1

            if (GPIO.input(21) == GPIO.HIGH and state == 0):
                counter += 1
                state = 1
                
            if (GPIO.input(21) == GPIO.LOW and state == 1):
                state = 0

        print('avg of',level,'is',sum/60)
        kit.motor1.throttle = 0
        kit.motor2.throttle = 0

    except:
        kit.motor1.throttle = 0
        kit.motor2.throttle = 0
    # en.start()
    # cn.start()