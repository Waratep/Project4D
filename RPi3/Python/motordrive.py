import time
from adafruit_motorkit import MotorKit

kit = MotorKit()

kit.motor1.throttle = 0.5
kit.motor2.throttle = 0.5

# time.sleep(0.5)
# kit.motor1.throttle = 0
# kit.motor2.throttle = 0