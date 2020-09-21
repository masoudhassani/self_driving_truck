import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from modules.utils import calibrate_camera
from modules.lane_detector import LaneDetector

# get all camera calibration images and calibrate the camera
# calibration = [ret, cMat, coefs, rvects, tvects] 
calbiration = calibrate_camera('resources/')

# instantiate a lane detector instance
detector = LaneDetector(calbiration)

frame = mpimg.imread('./resources/test_images/test4.jpg')
drawn_img = detector.detect(frame, show=False)
cv2.imwrite('./final.jpg', cv2.cvtColor(drawn_img, cv2.COLOR_RGB2BGR))

