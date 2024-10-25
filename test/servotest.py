from gpiozero import AngularServo
from time import sleep

from gpiozero.pins.pigpio import PiGPIOFactory

factory = PiGPIOFactory()

pan = AngularServo(14,min_angle = -180, max_angle = 180, min_pulse_width = 0.0005, max_pulse_width = 0.0025, pin_factory = factory)
tilt = AngularServo(15,min_angle = -90, max_angle = 90, min_pulse_width = 0.0005, max_pulse_width = 0.0025, pin_factory = factory)


pan.mid()
tilt.mid()

while True:
    pan.angle = -180
    tilt.angle = -90
    sleep(1)
    pan.angle = 0
    tilt.angle = 0
    sleep(1)
    pan.angle = 180
    tilt.angle = 90
    sleep(1)
    pan.angle = 0
    tilt.angle = 0
    sleep(2)
    