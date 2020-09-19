import numpy as np
import cv2
import glob
import os
import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

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
            raise Exception('error reading the calibration file')
    
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


# function to undistort a captured frame or image
def undistort(frame, cMat, coefs):
    if cMat is None: 
        raise Exception('no calibration found')
    
    return cv2.undistort(frame, cMat, coefs, None, cMat)

# receives a 1-channel image and returns the sobel bin
def get_sobel_bin(img):
    sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=9)  # x-direction gradient
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    sobel_bin = np.zeros_like(scaled_sobel)
    sobel_bin[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 1
    
    return sobel_bin

# convert undistorted frame to thresholed black and white
def get_threshold(img, show=False):
    ''' "img" should be an undistorted image ''' 
    
    # Color-space conversions
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    
    # Sobel gradient binaries
    sobel_s_bin = get_sobel_bin(s_channel)
    sobel_gray_bin = get_sobel_bin(gray)
    
    sobel_comb_bin = np.zeros_like(sobel_s_bin)
    sobel_comb_bin[(sobel_s_bin == 1) | (sobel_gray_bin == 1)] = 1
    
    # HLS S-Channel binary
    s_bin = np.zeros_like(s_channel)
    s_bin[(s_channel >= 150) & (s_channel <= 255)] = 1
    
    # Combine the binaries
    comb_bin = np.zeros_like(sobel_comb_bin)
    comb_bin[(sobel_comb_bin == 1) | (s_bin == 1)] = 1
    
    gray_img = np.dstack((gray, gray, gray))
    sobel_s_img = np.dstack((sobel_s_bin, sobel_s_bin, sobel_s_bin))*255
    sobel_gray_img = np.dstack((sobel_gray_bin, sobel_gray_bin, sobel_gray_bin))*255
    sobel_comb_img = np.dstack((sobel_comb_bin, sobel_comb_bin, sobel_comb_bin))*255
    s_img = np.dstack((s_bin, s_bin, s_bin))*255
    comb_img = np.dstack((comb_bin, comb_bin, comb_bin))*255
    
    if show: side_by_side_plot(img, comb_img, 'Original', 'Thresholded')
    
    return comb_img

def warp_to_lines(img, show=False):
    x_shape, y_shape = img.shape[1], img.shape[0]
    middle_x = x_shape//2
    top_y = 2*y_shape//3
    top_margin = 93
    bottom_margin = 450
    points = [
        (middle_x-top_margin, top_y),
        (middle_x+top_margin, top_y),
        (middle_x+bottom_margin, y_shape),
        (middle_x-bottom_margin, y_shape)
    ]

    '''
    # This shows the area we are warping to
    for i in range(len(points)):
        cv2.line(img, points[i-1], points[i], [255, 0, 0], 2)

    big_plot(img)
    '''

    src = np.float32(points)
    dst = np.float32([
        (middle_x-bottom_margin, 0),
        (middle_x+bottom_margin, 0),
        (middle_x+bottom_margin, y_shape),
        (middle_x-bottom_margin, y_shape)
    ])
    
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (x_shape, y_shape), flags=cv2.INTER_LINEAR)
    
    if show: side_by_side_plot(img, warped, 'Original', 'Warped Perspective')
        
    return warped, M, Minv

# helper functions for plotting
def side_by_side_plot(im1, im2, im1_title=None, im2_title=None):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    f.tight_layout()
    ax1.imshow(im1)
    if im1_title: ax1.set_title(im1_title, fontsize=30)
    ax2.imshow(im2)
    if im2_title: ax2.set_title(im2_title, fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    
def big_plot(img):
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.imshow(img)