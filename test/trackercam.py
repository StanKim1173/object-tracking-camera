import cv2
import numpy as np
from picamera2 import Picamera2

h_tolerance = 15;
s_tolerance = 50;

cam = Picamera2()
height = 480
width = 640
middle = (int(width / 2), int(height / 2))
lower = np.array([0,0,0])
upper = np.array([0,0,0])
lower2 = np.array([0,0,0])
upper2 = np.array([0,0,0])
hue = 0
sat = 0

area = 0
cam.configure(cam.create_video_configuration(main={"format": 'RGB888', "size": (width, height)}))

cam.start()

while True:
    frame = cv2.flip(cam.capture_array(),0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower, upper)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = mask1 + mask2
    cv2.circle(frame, middle, 15, (255, 0 , 255), 2)
    cv2.circle(mask, middle, 15, (255, 0 , 255), 2)
    mask_contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Finding contours in mask image

    # Finding position of all contours
    if len(mask_contours) != 0:
        area = 0
        for mask_contour in mask_contours:
            if cv2.contourArea(mask_contour) > 500:
                x, y, w, h = cv2.boundingRect(mask_contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3) #drawing rectangle
                if cv2.contourArea(mask_contour) > area:
                    x_max, y_max, w_max, h_max = x, y, w, h
                    area = cv2.contourArea(mask_contour)
        cv2.rectangle(frame, (x_max, y_max), (x_max + w_max, y_max + h_max), (255, 0, 255), 3) #drawing rectangle
        print("coordinates:", x_max + (w_max/2), y_max + (h_max/2))
    cv2.imshow('Default', frame)
    #cv2.imshow('HSV', hsv)
    cv2.imshow('Mask', mask)
    #print("center pixel RGB: ", frame[240,320,2], frame[240,320,1], frame[240,320,0])
    #print("center pixel HSV: ", hsv[240,320])
    #print("saved HS values: ", hue, sat)
    #print("masks: ", lower, upper, lower2, upper2)
    if cv2.waitKey(1) == ord('r'):
        hue = hsv[240,320,0]
        sat = hsv[240,320,1]
        lower = np.array([hue - h_tolerance, sat - s_tolerance, 0])
        upper = np.array([hue + h_tolerance, sat + s_tolerance, 255])
        lower2 = np.array([hue - h_tolerance, sat - s_tolerance, 0])
        upper2 = np.array([hue + h_tolerance, sat + s_tolerance, 255])
        
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
    if cv2.waitKey(1) == ord('q'):
        break
