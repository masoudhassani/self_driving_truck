import cv2
import numpy as np
import matplotlib.pyplot as plt
from modules.utils import *

# get all camera calibration images and calibrate the camera
ret, cMat, coefs, rvects, tvects = calibrate_camera('resources/')

# undistort the frame
frame = mpimg.imread('./resources/test_images/test1.jpg')
undistorted_frame = undistort(frame, cMat, coefs)

# create the threshold frame
threshold_img = get_threshold(undistorted_frame, show=False)
cv2.imwrite('./threshold_test2.jpg', threshold_img)

warped, M, Minv = warp_to_lines(threshold_img, show=False)
cv2.imwrite('./warped.jpg', warped)
