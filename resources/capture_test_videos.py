import numpy as np
import cv2
import argparse
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument('--system', type=str, default='pc')
parser.add_argument('--source', type=int, default=0)
parser.add_argument('--num_frames', type=int, default=120)
parser.add_argument('--file_name', type=str, default='test5.avi')
parser.add_argument('--fps', type=int, default=30)
args = parser.parse_args()

# settings
res = (640,480)
folder = './test_videos/'
if not os.path.exists(folder):
    os.makedirs(folder)
    
# initialize the camera
camera = cv2.VideoCapture(args.source)
time.sleep(1)

# Define the codec and create VideoWriter object
name = folder + args.file_name 
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(name,fourcc, args.fps, res)

counter = 1
print('starting to capture a video file')
while(camera.isOpened()):
    ret, frame = camera.read()
    if ret==True:
        #frame = cv2.flip(frame,0)

        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    
    if counter >= args.num_frames:
        break
    
    counter += 1

# Release everything if job is finished
camera.release()
out.release()
cv2.destroyAllWindows()