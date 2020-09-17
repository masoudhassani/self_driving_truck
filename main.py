import cv2
import numpy as np
import matplotlib.pyplot as plt
from modules.utils import calibrate_camera

# get all camera calibration images and calibrate the camera
ret, cMat, coefs, rvects, tvects = calibrate_camera('resources/')
print(ret, cMat, coefs, rvects, tvects)
print(type(coefs))