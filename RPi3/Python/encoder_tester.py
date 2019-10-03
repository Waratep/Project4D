from adafruit_motorkit import MotorKit
import RPi.GPIO as GPIO
import threading
import time
kit = MotorKit()

#const
circle = 21.5
circle_encoder = 20

left_encoder = 20
right_encoder = 21

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM) 
GPIO.setup(left_encoder, GPIO.IN)
GPIO.setup(right_encoder, GPIO.IN)


if __name__ == "__main__":

    left_counter_encoder = 0
    left_state_encoder = 0

    level = 0

    # try:

    kit.motor1.throttle = level
    millis = int(round(time.time() * 1000))
    while (1):

        if (int(round(time.time() * 1000)) - millis >= 1000):
            millis = int(round(time.time() * 1000))
            print('velocuty',(left_counter_encoder*circle)/circle_encoder,'cm/s')
            left_counter_encoder = 0

        if (GPIO.input(left_encoder) == GPIO.HIGH and left_state_encoder == 0):
            left_counter_encoder += 1
            left_state_encoder = 1
        if (GPIO.input(left_encoder) == GPIO.LOW and left_state_encoder == 1):
            left_state_encoder = 0


    kit.motor1.throttle = 0
    kit.motor2.throttle = 0

    # except:
    #     print('error')
    #     kit.motor1.throttle = 0
    #     kit.motor2.throttle = 0