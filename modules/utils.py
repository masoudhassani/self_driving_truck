import numpy as np
import cv2
import glob
import os
import pickle
import matplotlib.image as mpimg

# camera calibration function
def calibrate_camera(main_dir):
    # load the calibration file if exists
    if os.path.exists(main_dir + 'calibration.pkl'):
        try:
            with open(main_dir + 'calibration.pkl', 'rb') as fp:   # Unpickling
                lst = pickle.load(fp)
                
            ret = lst[0]
            cMat = lst[1]
            coefs = lst[2]
            rvects = lst[3]
            tvects = lst[4]
        
        except:
            print('error reading the calibration file')
    
    else:
        im_paths = glob.glob(main_dir+'camera_calibration_images/calibration*.jpg')
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
            with open(main_dir + 'calibration.pkl', 'wb') as fp:   # Unpickling
                lst = [ret, cMat, coefs, rvects, tvects]
                pickle.dump(lst, fp) 
                   
    return ret, cMat, coefs, rvects, tvects