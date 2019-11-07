from adafruit_motorkit import MotorKit
import RPi.GPIO as GPIO
import time


class Motor:

    def __init__(self):

        self.kit = MotorKit()

        self.circle_wheel = 21.5
        self.circle_encoder = 20
        self.left_encoder = 20
        self.right_encoder = 21
        self.left_counter_encoder = 0
        self.left_state_encoder = 0
        self.right_counter_encoder = 0
        self.right_state_encoder = 0
        self.speedLeftWheel = 0
        self.speedRightWheel = 0

        self.error = 0
        self.lasterror = 0
        self.output = 0
        self.sumerror = 0

        self.kp = 2
        self.ki = 0.0
        self.kd = 0.0

        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM) 
        GPIO.setup(self.left_encoder, GPIO.IN)
        GPIO.setup(self.right_encoder, GPIO.IN)

        self.millis = int(round(time.time() * 1000))

    def map(self, x,  in_min,  in_max,  out_min,  out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


    def callbackleft_encoder(self,dummy):
        self.left_counter_encoder += 1


    def callbackright_encoder(self,dummy):
        self.right_counter_encoder += 1

    
    def run(self,debug = False):

        if (int(round(time.time() * 1000)) - self.millis >= 100):
            self.millis = int(round(time.time() * 1000))
            self.speedLeftWheel = self.left_counter_encoder*self.circle_wheel/self.circle_encoder * 10
            self.speedRightWheel = self.right_counter_encoder*self.circle_wheel/self.circle_encoder * 10
            self.left_counter_encoder = 0
            self.right_counter_encoder = 0
            
            if(debug):
                print('Velocity Motor Rihgt',"{:2f}".format(self.speedRightWheel),'cm/s')
            

    def stop(self):
        self.kit.motor2.throttle = 0
        self.kit.motor1.throttle = 0

    def setMotorLeft(self,pwm):
        if(pwm > 1):
            pwm = 1
        if(pwm < -1):
            pwm = -1
        self.kit.motor1.throttle = pwm
    def setMotorRight(self,pwm):
        if(pwm > 1):
            pwm = 1
        if(pwm < -1):
            pwm = -1
        self.kit.motor2.throttle = pwm


    def setMotor(self,speed,debug = False):

        self.lasterror = self.error 
        self.error = speed - self.speedRightWheel
        self.sumerror += self.error

        self.output = ( self.error * self.kp ) + ( self.sumerror * self.ki ) + ( ( self.error - self.lasterror ) * self.kd )

        if(self.output > 100): 
            self.output = 100
        if(self.output < -100):
            self.output = -100

        if(debug):
            print('self.output',"{:2f}".format(self.output),'self.speedRightWheel',"{:2f}".format(self.speedRightWheel),'self.error',"{:2f}".format(self.error))
        
        self.kit.motor2.throttle = self.output / 100
        # self.kit.motor1.throttle = self.output / 100

    def getMotorTurnLeft(self):
        return self.speedLeftWheel

    def getMotorTurnRight(self):
        return self.speedRightWheel    
