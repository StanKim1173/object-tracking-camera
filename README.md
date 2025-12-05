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

## What did I personally do and what was not my work?
A sizeable portion of code that is used in this project (and the object recognition models used) are from publicly available sources, which I have linked to below.
What I personally worked on were the following:
- A very simple GUI overlaying the video output for user controls
- All of the keyboard inputs used to control the camera
- The logic used to control the servos to turn the camera towards the object
- The ability to detect which color/object was in the center of the screen and saving that information so that the camera knows what to focus on
- The ability to save a color or object in real time in order to track it
- Wiring the servos and camera to the Raspberry Pi

What I modified were the following:
- The detection and highlighting of the largest bounding box for a given color/object
- The camera settings, particularly the video resolution in order to achieve a balance between picture quality and framerate

What I did not personally make:
- The TensorFlow model itself, this was obtained from one of the resources linked below
- The basic setup to get OpenCV running

## What more could I have done with this?
Originally, I was planning for this to a component to a wheeled robot which I can remotely control through a PC or other device. However, this would have been a much harder task to complete. Having taken a class at my university in which I did build a robot in a group of 8 students (which you can see the results of here: https://sites.google.com/terpmail.umd.edu/clyde-otv/home), I was already aware that doing all of this on my own would have been incredibly time consuming and expensive, which is why I decided not to continue further. If I were to have built the full robot, this would have required that I:
- Spend even more money on all of the components
- Design a chassis for the robot
- Work on motor control software
- Find or develop software to allow remote controlling of the robot and possibly stream the video feed to another device
- Identify a portable power source that would be sufficient enough to power the Raspberry Pi and the motors


## Demonstration
Video demonstration can be found here: https://youtu.be/95bli5oYAKs?si=FGds9GNFk_Q60r-o

## References
Listed below are the resources that I took inspiration from or used as a basis.
- https://github.com/computervisioneng/color-detection-opencv
- https://core-electronics.com.au/guides/object-identify-raspberry-pi/
