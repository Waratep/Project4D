
import RPi.GPIO as IO          
import time                            

IO.setwarnings(False)           
IO.setmode (IO.BCM)        
IO.setup(19,IO.OUT)           
IO.setup(13,IO.OUT)           

p = IO.PWM(19,100)          
p2 = IO.PWM(13,100)          

p.start(0)                              
p2.start(0)                              

while 1:                              

    for x in range (50):                          
        p.ChangeDutyCycle(x)   
        p2.ChangeDutyCycle(x)              
        time.sleep(0.1)                          
    for x in range (50):                        
        p.ChangeDutyCycle(50-x)    
        p2.ChangeDutyCycle(50-x)       
        time.sleep(0.1)                         
