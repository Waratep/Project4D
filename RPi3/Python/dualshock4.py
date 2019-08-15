from asyncore import file_dispatcher, loop
from evdev import InputDevice, categorize, ecodes
import time

dev = InputDevice('/dev/input/event0')


def read_button(data):
    for event in data.read_loop():
        if event.type == ecodes.EV_KEY:
            btn = categorize(event).keycode
            if btn[0] == 'BTN_B':
                print("X")
            if btn == 'BTN_C':
                print("O")
            if btn[0] == 'BTN_A':
                print("#")
            if btn[0] == 'BTN_NORTH':
                print("/\\")
            if btn == 'BTN_Z':
                print("R1")
            if btn == 'BTN_TR':
                print("R2",event.value)
            if btn[0] == 'BTN_WEST':
                print("L1")
            if btn == 'BTN_TL':
                print("L2",event.value)
            if btn == 'BTN_TR2':
                print("Option")
            if btn == 'BTN_TL2':
                print("Share")
            if btn == 'BTN_MODE':
                print("Mode")
            if btn == 'BTN_THUMBL':
                print("Click")
            if btn == 'BTN_START':
                print("Click_Right")
            if btn == 'BTN_SELECT':
                print("Click_Left")


def read_axes(data):
    speed = 0
    for event in data.read_loop():
        if (event.type == ecodes.EV_ABS):
            if event.code == 3: #break_left_motor
                print(event.value)
                pass

            if event.code == 4: #break_right_motor
                print(event.value)
                pass


try:
    while 1:

        read_axes(dev)
        # read_button(dev)


except KeyboardInterrupt:
    print("ERROR")
    



















