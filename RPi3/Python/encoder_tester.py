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
        self.ki = 1
        self.kd = 0

        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM) 
        GPIO.setup(self.left_encoder, GPIO.IN)
        GPIO.setup(self.right_encoder, GPIO.IN)

        self.millis = int(round(time.time() * 1000))

    def map(self, x,  in_min,  in_max,  out_min,  out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


    def callbackleft_encoder(self,dummy):
        # if (GPIO.input(self.left_encoder) == GPIO.HIGH and self.left_state_encoder == 0):
        self.left_counter_encoder += 1
        print('right',self.right_counter_encoder,'left',self.left_counter_encoder)

        #     self.left_state_encoder = 1
        # if (GPIO.input(self.left_encoder) == GPIO.LOW and self.left_state_encoder == 1):
        #     self.left_state_encoder = 0

    def callbackright_encoder(self,dummy):
        # if (GPIO.input(self.right_encoder) == GPIO.HIGH and self.right_state_encoder == 0):
        self.right_counter_encoder += 1
        print('right',self.right_counter_encoder,'left',self.left_counter_encoder)
        #     self.right_state_encoder = 1
        # if (GPIO.input(self.right_encoder) == GPIO.LOW and self.right_state_encoder == 1):
        #     self.right_state_encoder = 0
    
    def run(self):

        if (int(round(time.time() * 1000)) - self.millis >= 1000):
            self.millis = int(round(time.time() * 1000))
            self.speedLeftWheel = self.left_counter_encoder*self.circle_wheel/self.circle_encoder
            self.speedRightWheel = self.right_counter_encoder*self.circle_wheel/self.circle_encoder
            self.left_counter_encoder = 0
            self.right_counter_encoder = 0
            
            # print('Velocity Motor Left', self.speedLeftWheel,'cm/s')
            print('Velocity Motor Rihgt',self.speedRightWheel,'cm/s')
            

    def stop(self):
        pass

    def setMotor(self,speed):
        self.lasterror = self.error 
        self.error = speed - self.speedRightWheel
        self.sumerror += self.error
        self.output = ( self.error * self.kp ) + ( self.sumerror * self.ki ) + ( ( self.error - self.lasterror ) * self.kd )

        if(self.output > 100): 
            self.output = 100
        if(self.output < -100):
            self.output = -100

        print('self.output',self.output,'self.speedRightWheel',self.speedRightWheel,'self.error',self.error)
        self.kit.motor2.throttle = self.output / 100
        self.kit.motor1.throttle = self.output / 100

    def getMotorTurnLeft(self):
        return self.speedLeftWheel

    def getMotorTurnRight(self):
        return self.speedRightWheel    
    

if __name__ == "__main__":

    motor = Motor()
 
    GPIO.add_event_detect(motor.left_encoder, GPIO.FALLING, motor.callbackleft_encoder, bouncetime=1)  
    GPIO.add_event_detect(motor.right_encoder, GPIO.FALLING, motor.callbackright_encoder, bouncetime=1)  

    try: 
    
        while (1): 
            
            # motor.run()
            # motor.setMotor(0)

            time.sleep(0.1)

    except KeyboardInterrupt:
        print('KeyboardInterrupt')