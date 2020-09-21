import numpy as np
import cv2
from threading import Thread
import pickle
import os
import imutils

class Camera:
    def __init__(self, source=1, frame_width=1280, fps=120, path='/'):

        # for the elp camera we use DSHOW instead of default MSMF (microsoft media foundation)
        if type(source) is str:
            self.video_file = True
            self.cap = cv2.VideoCapture(source)
            self.frame_width = frame_width
        else:
            self.video_file = False
            self.cap = cv2.VideoCapture(source+cv2.CAP_DSHOW) 
        
        _, self.frame = self.cap.read()
        self.aspect_ratio = self.frame.shape[1] / self.frame.shape[0]   # width/height
        self.stopped = False

        if not self.video_file:
            # set camera properties
            self.cap.set(3, frame_width)            # width
            self.cap.set(4, frame_width / self.aspect_ratio)   # height, elp camera has 16/9 ratio
            self.cap.set(5,fps)           # frames per second

            self.frame_width = int(self.cap.get(3))   
            self.frame_height = int(self.cap.get(4)) 

            print("fps:", int(self.cap.get(5)))
            print('width:', self.frame_width)
            print('height:', self.frame_height)

        self.kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])   
        self.path = path   
        
        # start the camera thread
        self.start()

    def pre_processing(self, img):
        processed_frame = cv2.filter2D(img, -1, self.kernel)
        return processed_frame

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()

    def update(self):
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            _, self.frame = self.cap.read()

    def return_frame(self):
        # return the frame most recently read
        if self.video_file and self.frame is not None:
            self.frame = imutils.resize(self.frame, width=self.frame_width)
            
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        self.cap.release()
        
    # camera calibration function
    def calibrate(self):
        # load the calibration file if exists
        if os.path.exists(self.path + 'calibration.pkl'):
            try:
                with open(self.path + 'calibration.pkl', 'rb') as fp:   # Unpickling
                    lst = pickle.load(fp)
                    
                ret = lst[0]
                cMat = lst[1]
                coefs = lst[2]
                rvects = lst[3]
                tvects = lst[4]
            
            except:
                raise Exception('error reading the calibration file')
        
        else:
            im_paths = glob.glob(self.path+'camera_calibration_images/calibration*.jpg')
            cb_shape = (9, 6)  # Corners we expect to be detected on the chessboard

            obj_points = []  # 3D points in real-world space
            img_points = []  # 2D points in the image

            for im_path in im_paths:
                img = mpimg.imread(im_path)

                obj_p = np.zeros((cb_shape[0]*cb_shape[1], 3), np.float32)
                coords = np.mgrid[0:cb_shape[0], 0:cb_shape[1]].T.reshape(-1, 2)  # x, y coords

                obj_p[:,:2] = coords

                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                found_all, corners = cv2.findChessboardCorners(gray, cb_shape, None)
                if found_all:
                    obj_points.append(obj_p)
                    img_points.append(corners)
                    img = cv2.drawChessboardCorners(img, cb_shape, corners, found_all)
                else:
                    print("Couldn't find all corners in image:", im_path)
        
                ret, cMat, coefs, rvects, tvects = cv2.calibrateCamera(obj_points, img_points, gray.shape, None, None)
                
                # save the calibration file
                with open(self.path + 'calibration.pkl', 'wb') as fp:   # Unpickling
                    lst = [ret, cMat, coefs, rvects, tvects]
                    pickle.dump(lst, fp) 
                    
        self.calibration = [ret, cMat, coefs, rvects, tvects]  
        return self.calibration