# Object Tracking Camera
This camera allows the user to move a camera around, identify an object based on its color or physical appearance (using TensorFlowLite for ML-based recognition), and automatically follow its position.

## Project Details
This project was done on a Raspberry Pi 4 running the Raspberry Pi OS, which is based on the Debian Linux distribution.
It is connected to a Raspberry Pi V2 Camera which is glued onto two cheap hobby servos in a fashion that allows the camera to pan up, down, left and right.
The software used to control the camera is written in Python, and makes use of the OpenCV library for video and image functionality. 


## Why did I do this?
When I set out to do this project, I wanted to explore some of the tools used by hobbyists on the internet who were making cool things and posting them online.
While there are many things I learned from my education, I felt like I did not get that many opportunities to apply some of the knowledge that I had gained.
After finding some projects online similar to this one, I decided that I wanted to try building this project myself and see what it takes to develop a product that might actually have some use.

## What did I personally do?
A sizeable portion of code that is used in this project (and the object recognition models used) are from publicly available sources, which I have linked to below.
What I personally worked on were the following:
- A very simple GUI overlaying the video output for user controls
- All of the keyboard inputs used to control the camera
- The logic used to control the servos to turn the camera towards the object
- The ability to detect which color/object was in the center of the screen and saving that information so that the camera knows what to focus on

What I modified were the following:
- The detection and drawing of bounding boxes around objects
- The general setup for the camera
- The TensorFlow implementation of ML-based object recognition - this was essentially a drop-in solution

## Demonstration
Video demonstration can be found here: https://youtu.be/95bli5oYAKs?si=FGds9GNFk_Q60r-o

## References
Listed below are the resources that I took inspiration from or used as a basis.
- https://github.com/computervisioneng/color-detection-opencv
- https://core-electronics.com.au/guides/object-identify-raspberry-pi/
