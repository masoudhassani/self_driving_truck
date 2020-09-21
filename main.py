import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from modules.lane_detector import LaneDetector
from modules.camera import Camera

RENDER = True
# instantiate a camera
# get all camera calibration images and calibrate the camera
# calibration = [ret, cMat, coefs, rvects, tvects] 
camera = Camera(source='resources/test_videos/test2.mp4',
                frame_width=640,
                fps=60,
                path='resources/')
calibration = camera.calibrate() 

# instantiate a lane detector instance
detector = LaneDetector(calibration)

# read camera or video file and detect lanes
counter = 0
t_start = time.time()
max_time_steps = 60

while 1:
    t = time.time()

    # Capture frame-by-frame
    frame = camera.return_frame()

    if frame is not None:
        frame_with_lanes = detector.detect(frame, show=False)
        
        if RENDER:
            # Display the resulting frame
            cv2.imshow('frame with lanes',frame_with_lanes)

    counter += 1
    dt = time.time() - t
    print(dt)

    if counter > max_time_steps:  
        camera.stop()   
        fps = int(counter/(time.time() - t_start))
        print('real fps achieved:', fps)
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        # When everything done, release the capture
        camera.stop()  
        cv2.destroyAllWindows()
        break


