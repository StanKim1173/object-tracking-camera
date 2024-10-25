import numpy as np
import cv2 as cv
 
lower = np.array([15, 150, 20])
upper = np.array([35, 255, 255])
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
 
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    image = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(image, lower, upper)
    # Display the resulting frame
    cv.imshow('frame', image)
    cv.imshow('mask', mask)
    if cv.waitKey(1) == ord('q'):
        break
 
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()