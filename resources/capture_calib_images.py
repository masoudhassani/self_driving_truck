import cv2
import argparse
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument('--system', type=str, default='pc')
parser.add_argument('--source', type=int, default=0)
parser.add_argument('--num_captures', type=int, default=20)
args = parser.parse_args()
 
# important settings 
res = (1280, 960)

# create folder to store pictures
folder = './camera_calibration_images'
if not os.path.exists(folder):
    os.makedirs(folder)

camera = cv2.VideoCapture(args.source)
camera.set(cv2.CAP_PROP_FRAME_WIDTH,res[0])
camera.set(cv2.CAP_PROP_FRAME_HEIGHT,res[1])

for i in range(args.num_captures):
    print('capturing {} from {} in 2 second'.format(i, args.num_captures))
    time.sleep(2)
    
    ret, img = camera.read()
    name = folder+'/calibration'+str(i)+'.jpg'
    if ret:
        print('captured')
        cv2.imwrite(name, img)
        
    time.sleep(1)
        
# Release everything if job is finished
camera.release()