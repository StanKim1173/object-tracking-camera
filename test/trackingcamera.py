import os
#enables pigpio library through command line call
os.system('sudo pigpiod')
import cv2
import time
import numpy as np
from picamera2 import Picamera2
from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory

#creates instantiation of pigpio
factory = PiGPIOFactory()
#initializes pan and tilt servos and sets them to default position
pan = AngularServo(14,min_angle = -180, max_angle = 180, min_pulse_width = 0.0005, max_pulse_width = 0.0025, pin_factory = factory)
tilt = AngularServo(15,min_angle = -180, max_angle = 180, min_pulse_width = 0.0005, max_pulse_width = 0.0025, pin_factory = factory)
pan.mid()
tilt.mid()

# Variable control determines how the camera is controlled. The following settings are:
# 0 - Manual control, using WASD to control the pan and tilt of the camera
# 1 - Color-based control, uses selected color to control the pan and tilt
control = 1
control_type = "Automatic - Color"
#defines +/- range for hue and saturation values for valid colors
h_tolerance = 15;
s_tolerance = 50;
#defining resolution
width = 960
height = 720
middle = (int(width / 2), int(height / 2))
x_center = int(width / 2)
y_center = int(height / 2)
#initializes camera
cam = Picamera2()
config = cam.create_preview_configuration(raw={'size': (width, height)})
cam.configure(config)
# Variables for color selection. There are two arrays each for lower and upper bound because the way
# that hue value can wrap around from 0 and 255 and vise versa, so we need two arrays to account for
# the wrap around.
lower = np.array([0,0,0])
upper = np.array([0,0,0])
lower2 = np.array([0,0,0])
upper2 = np.array([0,0,0])
hue = 0
sat = 0
area = 0
rgb = (0,0,0)
instructions = 0

fps = 0
t_start = 0

font = cv2.FONT_HERSHEY_DUPLEX

cam.configure(cam.create_video_configuration(main={"format": 'RGB888', "size": (width, height)}))
cam.start()

while True:
    #sets frame as main display; image is flipped because the camera is mounted upside down
    frame = cv2.flip(cam.capture_array(),0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Combines the two color masks together into one (why there are two masks is explained above)
    mask1 = cv2.inRange(hsv, lower, upper)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = mask1 + mask2
    # Draws circle in middle of screen
    cv2.circle(frame, middle, 15, (255, 0 , 255), 2)
    
    # Finding position of all contours
    mask_contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Finding contours in mask image
    if len(mask_contours) != 0:
        area = 0
        is_there = 0
        for mask_contour in mask_contours:
            if cv2.contourArea(mask_contour) > 200:
                is_there = 1
                x, y, w, h = cv2.boundingRect(mask_contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2) #drawing rectangle
                if cv2.contourArea(mask_contour) > area:
                    x_max, y_max, w_max, h_max = x, y, w, h
                    area = cv2.contourArea(mask_contour)
        if is_there == 1:
            cv2.rectangle(frame, (x_max, y_max), (x_max + w_max, y_max + h_max), (255, 0, 255), 2) #drawing rectangle
            #sets x and y coordinates of largest object
            x_coord, y_coord = x_max + (w_max/2), y_max + (h_max/2)
            #debug print statement
            print("coordinates:", x_coord, y_coord)
            #controls the servos that move the camera for object tracking
            #integer division allows for more drastic movement
            if control == 1:
                if x_coord < x_center - 20 and pan.angle > -175:
                    pan.angle -= 1 + ((x_center - x_coord) // 80)
                elif x_coord > x_center + 20 and pan.angle < 175:
                    pan.angle += 1 + ((x_coord - x_center) // 80)
                if y_coord < y_center - 20 and tilt.angle < 60:
                    tilt.angle += 1 + ((y_center - y_coord) // 100)
                elif y_coord > y_center + 20 and tilt.angle > -60:
                    tilt.angle -= 1 + ((y_coord - y_center) // 100)
    t_end = time.time()    
    fps = int(1/(t_end - t_start))
    t_start = t_end
    cv2.putText(frame, f"FPS: {fps}", (10, 50), font, 0.5, (255,255,255), 1, 2)
    cv2.putText(frame, f"Resolution: {width}x{height}", (10, 20), font, 0.5, (255,255,255), 1, 2)
    cv2.putText(frame, f"Control Type: {control_type}", (10, height - 10), font, 1, (255,255,255), 1, 2)
    cv2.putText(frame, "Target Color", (width - 210, height - 15), font, 1, rgb , 1, 2)
    cv2.putText(frame, "Toggle Instructions:", (width - 400, 30), font, 1, (255,255,255), 1, 2)
    cv2.putText(frame, "H", (width - 30, 30), font, 1, (255,255,255), 1, 2)
    
    if instructions == 1:
        cv2.putText(frame, "Quit:", (width - 400, 60), font, 1, (255,255,255), 1, 2)
        cv2.putText(frame, "Q", (width - 30, 60), font, 1, (255,255,255), 1, 2)
        cv2.putText(frame, "Toggle Mode:", (width - 400, 90), font, 1, (255,255,255), 1, 2)
        cv2.putText(frame, "M", (width - 30, 90), font, 1, (255,255,255), 1, 2)
        cv2.putText(frame, "Select Color:", (width - 400, 120), font, 1, (255,255,255), 1, 2)
        cv2.putText(frame, "F", (width - 30, 120), font, 1, (255,255,255), 1, 2)
        cv2.putText(frame, "Reset Color:", (width - 400, 150), font, 1, (255,255,255), 1, 2)
        cv2.putText(frame, "R", (width - 30, 150), font, 1, (255,255,255), 1, 2)
        if control == 0:
            cv2.putText(frame, "Manual Controls:", (width - 400, 180), font, 1, (255,255,255), 1, 2)
            cv2.putText(frame, "Pan Left:", (width - 300, 210), font, 1, (255,255,255), 1, 2)
            cv2.putText(frame, "A", (width - 30, 210), font, 1, (255,255,255), 1, 2)
            cv2.putText(frame, "Pan Right:", (width - 300, 240), font, 1, (255,255,255), 1, 2)
            cv2.putText(frame, "D", (width - 30, 240), font, 1, (255,255,255), 1, 2)
            cv2.putText(frame, "Tilt Up:", (width - 300, 270), font, 1, (255,255,255), 1, 2)
            cv2.putText(frame, "W", (width - 30, 270), font, 1, (255,255,255), 1, 2)
            cv2.putText(frame, "Tilt Down:", (width - 300, 300), font, 1, (255,255,255), 1, 2)
            cv2.putText(frame, "S", (width - 30, 300), font, 1, (255,255,255), 1, 2)
            cv2.putText(frame, "Center:", (width - 300, 330), font, 1, (255,255,255), 1, 2)
            cv2.putText(frame, "X", (width - 30, 330), font, 1, (255,255,255), 1, 2)
    
    cv2.imshow('Default', frame)
    #cv2.imshow('HSV', hsv)
    cv2.imshow('Mask', mask)
    #print("center pixel RGB: ", frame[240,320,2], frame[240,320,1], frame[240,320,0])
    #print("center pixel HSV: ", hsv[240,320])
    #print("saved HS values: ", hue, sat)
    #print("masks: ", lower, upper, lower2, upper2)
    
    
    # Remaining code are keyboard input based for camera control. Controls are as follows:
    # Q - quits program
    # F - sets the target color to the color located in the center of the image (320,240)
    # R - resets the target color to default values
    # M - toggles the control mode
        # W,A,S,D - moves the camera up, left, down, and right if control is set to 0 (manual)
        # X - resets camera position to default
    # H - toggles on-screen instructions
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('f'):
        hue = hsv[y_center,x_center,0]
        sat = hsv[y_center,x_center,1]
        rgb = (int(frame[y_center,x_center,0]),int(frame[y_center,x_center,1]),int(frame[y_center,x_center,2]))
        print(rgb)
        lower = np.array([hue - h_tolerance, sat - s_tolerance, 0])
        upper = np.array([hue + h_tolerance, sat + s_tolerance, 255])
        lower2 = np.array([hue - h_tolerance, sat - s_tolerance, 0])
        upper2 = np.array([hue + h_tolerance, sat + s_tolerance, 255])
        # When designating a range for hue and saturation, they may go beyond the lower and upper limits.
        # If so, we need to loop back to the other side and continue our range, which is what the code
        # below is intended for.
        if lower[0] < 0:
            lower[0] = 0
            lower2[0] = lower2[0] + 179
            upper2[0] = 179
        elif upper[0] > 179:
            upper[0] = upper[0] - 179
            upper2[0] = 179
            lower2[0] = 0
        if lower[1] < 0:
            lower[1] = 0
            lower2[1] = 0
        if upper[1] > 255:
            upper[1] = 255
            upper2[1] = 255
    elif key == ord('r'):
        lower = np.array([0,0,0])
        upper = np.array([0,0,0])
        lower2 = np.array([0,0,0])
        upper2 = np.array([0,0,0])
        hue = 0
        sat = 0
        area = 0
        rgb = (0,0,0)
    elif key == ord('m'):
        if control == 1:
            control = 0
            control_type = "Manual"
        else:
            control = 1
            control_type = "Automatic - Color"
    elif key == ord('h'):
        if instructions == 0:
            instructions = 1
            print("Instructions ON")
        else:
            instructions = 0
            print("Instructions OFF")
    elif control == 0:
        if key == ord('w'):
            if tilt.angle < 60:
                tilt.angle += 1
        elif key == ord('s'):
            if tilt.angle > -60:
                tilt.angle -= 1
        elif key == ord('a'):
            if pan.angle < 175:
                pan.angle += 1
        elif key == ord('d'):
            if pan.angle > -175:
                pan.angle -= 1
        elif key == ord('x'):
            pan.mid()
            tilt.mid()

cv2.destroyAllWindows()
