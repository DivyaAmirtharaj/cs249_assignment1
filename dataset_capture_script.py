# Dataset Capture Script - By: divyaamirtharaj - Tue Oct 3 2023

# Use this script to control how your OpenMV Cam captures images for your dataset.
# You should apply the same image pre-processing steps you expect to run on images
# that you will feed to your model during run-time.

import sensor, image, time
import pyb

sensor.reset()
sensor.set_pixformat(sensor.RGB565) # Modify as you like.
sensor.set_framesize(sensor.QVGA) # Modify as you like.
sensor.skip_frames(time = 2000)
redLED = pyb.LED(1) # built-in red LED
blueLED = pyb.LED(3) # built-in blue LED

clock = time.clock()
n = 0

while(True):
    clock.tick()
    img = sensor.snapshot()
    print(clock.fps())
