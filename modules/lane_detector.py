import numpy as np
import cv2
import glob
import os
import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from .lane import Lane
from .utils import undistort, get_threshold, warp, big_plot, side_by_side_plot

class LaneDetector:
    def __init__(self, calibration):
        self.calibration = calibration
        
        # some internal variables
        self.n_lanes_to_keep = 5
        self.prev_lanes = []
        self.n_bad_lanes = 0
        
    # fine lanes in a frame 
    def find_lane(self, warped, show=False):
        # Create a binary version of the warped image
        warped_bin = np.zeros_like(warped[:,:,0])
        warped_bin[(warped[:,:,0] > 0)] = 1
        
        vis_img = warped.copy()  # The image we will draw on to show the lane-finding process
        vis_img[vis_img > 0] = 255  # Max out non-black pixels so we can remove them later

        # Sum the columns in the bottom portion of the image to create a histogram
        histogram = np.sum(warped_bin[warped_bin.shape[0]//2:,:], axis=0)
        # Find the left an right right peaks of the histogram
        midpoint = histogram.shape[0]//2
        left_x = np.argmax(histogram[:midpoint])  # x-position for the left window
        right_x = np.argmax(histogram[midpoint:]) + midpoint  # x-position for the right window

        n_windows = 10
        win_height = warped_bin.shape[0]//n_windows
        margin = 50  # Determines how wide the window is
        pix_to_recenter = margin*2  # If we find this many pixels in our window we will recenter (too few would be a bad recenter)

        # Find the non-zero x and y indices
        nonzero_ind = warped_bin.nonzero()
        nonzero_y_ind = np.array(nonzero_ind[0])
        nonzero_x_ind = np.array(nonzero_ind[1])

        left_line_ind, right_line_ind = [], []

        for win_i in range(n_windows):
            win_y_low = warped_bin.shape[0] - (win_i+1)*win_height
            win_y_high = warped_bin.shape[0] - (win_i)*win_height
            win_x_left_low = max(0, left_x - margin)
            win_x_left_high = left_x + margin
            win_x_right_low = right_x - margin
            win_x_right_high = min(warped_bin.shape[1]-1, right_x + margin)

            # Draw the windows on the vis_img
            rect_color, rect_thickness = (0, 255, 0), 3
            cv2.rectangle(vis_img, (win_x_left_low, win_y_high), (win_x_left_high, win_y_low), rect_color, rect_thickness)
            cv2.rectangle(vis_img, (win_x_right_low, win_y_high), (win_x_right_high, win_y_low), rect_color, rect_thickness)

            # Record the non-zero pixels within the windows
            left_ind = (
                (nonzero_y_ind >= win_y_low) &
                (nonzero_y_ind <= win_y_high) &
                (nonzero_x_ind >= win_x_left_low) &
                (nonzero_x_ind <= win_x_left_high)
            ).nonzero()[0]
            right_ind = (
                (nonzero_y_ind >= win_y_low) &
                (nonzero_y_ind <= win_y_high) &
                (nonzero_x_ind >= win_x_right_low) &
                (nonzero_x_ind <= win_x_right_high)
            ).nonzero()[0]
            left_line_ind.append(left_ind)
            right_line_ind.append(right_ind)

            # If there are enough pixels, re-align the window
            if len(left_ind) > pix_to_recenter:
                left_x = int(np.mean(nonzero_x_ind[left_ind]))
            if len(right_ind) > pix_to_recenter:
                right_x = int(np.mean(nonzero_x_ind[right_ind]))

        # Combine the arrays of line indices
        left_line_ind = np.concatenate(left_line_ind)
        right_line_ind = np.concatenate(right_line_ind)

        # Gather the final line pixel positions
        left_x = nonzero_x_ind[left_line_ind]
        left_y = nonzero_y_ind[left_line_ind]
        right_x = nonzero_x_ind[right_line_ind]
        right_y = nonzero_y_ind[right_line_ind]

        # Color the lines on the vis_img
        vis_img[left_y, left_x] = [254, 0, 0]  # 254 so we can isolate the white 255 later
        vis_img[right_y, right_x] = [0, 0, 254]  # 254 so we can isolate the white 255 later
        
        # Fit a 2nd-order polynomial to the lines
        left_fit = np.polyfit(left_y, left_x, 2)
        right_fit = np.polyfit(right_y, right_x, 2)

        # Get our x/y vals for the fit lines
        y_vals = np.linspace(0, warped_bin.shape[0]-1, warped_bin.shape[0])
        left_x_vals = left_fit[0]*y_vals**2 + left_fit[1]*y_vals + left_fit[2]
        right_x_vals = right_fit[0]*y_vals**2 + right_fit[1]*y_vals + right_fit[2]
        
        # Calculate real-world curvature for each lane line
        left_curvature = self.calc_curvature(left_y, left_x, np.max(y_vals))
        right_curvature = self.calc_curvature(right_y, right_x, np.max(y_vals))
        # Calculate the center-lane offset of the car
        offset = self.calc_offset(left_x_vals[-1], right_x_vals[-1], warped.shape[1]//2)

        if show:
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.imshow(vis_img)
            ax.plot(left_x_vals, y_vals, color='yellow')
            ax.plot(right_x_vals, y_vals, color='yellow')
            
        lane_lines_img = vis_img.copy()
        lane_lines_img[lane_lines_img == 255] = 0  # This basically removes everything except the colored lane lines
        
        # Build the Lane object to return
        lane = Lane()
        lane.y_vals = y_vals
        lane.left_x_vals = left_x_vals
        lane.right_x_vals = right_x_vals
        lane.lane_lines_img = lane_lines_img
        lane.left_curvature = left_curvature
        lane.right_curvature = right_curvature
        lane.offset = offset
        
        return lane

    def draw_values(self, img, lane, show=False):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = .7
        color =  (0, 0, 0)
        line_type = cv2.LINE_AA
        cv2.putText(
            img,
            'Left, Right Curvature: {:.2f}m, {:.2f}m'.format(lane.left_curvature, lane.right_curvature),
            (20, 40),  # origin point
            font,
            scale,
            color,
            lineType=line_type
        )
        cv2.putText(
            img,
            'Center-Lane Offset: {:.2f}m'.format(lane.offset),
            (20, 80),  # origin point
            font,
            scale,
            color,
            lineType=line_type
        )
        
        if show: 
            big_plot(img)
        
        return img
        
    def draw_lane(self, img, lane, Minv, show=False):
        # Prepare the x/y points for cv2.fillPoly()
        left_points = np.array([np.vstack([lane.left_x_vals, lane.y_vals]).T])
        right_points = np.array([np.flipud(np.vstack([lane.right_x_vals, lane.y_vals]).T)])
        # right_points = np.array([np.vstack([right_x_vals, y_vals]).T])
        points = np.hstack((left_points, right_points))

        # Color the area between the lines (the lane)
        filled_lane = np.zeros_like(lane.lane_lines_img)  # Create a blank canvas to draw the lane on
        cv2.fillPoly(filled_lane, np.int_([points]), (0, 255, 0))
        warped_lane_info = cv2.addWeighted(lane.lane_lines_img, 1, filled_lane, .3, 0)

        unwarped_lane_info = cv2.warpPerspective(warped_lane_info, Minv, (img.shape[1], img.shape[0]))
        drawn_img = cv2.addWeighted(img, 1, unwarped_lane_info, 1, 0)
        drawn_img = self.draw_values(drawn_img, lane)
        
        if show: 
            big_plot(drawn_img)
            
        return drawn_img

    def calc_curvature(self, y_to_fit, x_to_fit, y_eval):
        # Conversion factors for pixels to meters
        m_per_pix_y, m_per_pix_x = 30/720, 3.7/700
        
        # Fit a new polynomial to world-space (in meters)
        fit = np.polyfit(y_to_fit*m_per_pix_y, x_to_fit*m_per_pix_x, 2)
        curvature = ((1 + (2*fit[0]*(y_eval*m_per_pix_y) + fit[1])**2)**1.5) / np.absolute(2*fit[0])
        return curvature

    def calc_offset(self, left_x, right_x, img_center_x):
        lane_width = abs(left_x - right_x)
        lane_center_x = (left_x + right_x)//2
        pix_offset = img_center_x - lane_center_x
        
        lane_width_m = 3.7  # How wide we expect the lane to be in meters
        return lane_width_m * (pix_offset/lane_width)

    # filtering - Tests whether a new lane is appropriate compared to previously found lanes
    def is_good_lane(self, lane):       
        good_line_diff = True
        good_lane_area = True
        
        # Measure the total x-pixel difference between the new and the old
        if len(self.prev_lanes) > 0:  # If we don't have any previous lanes yet, just assume True
            prev_x_left = self.prev_lanes[0].left_x_vals
            prev_x_right = self.prev_lanes[0].right_x_vals
            current_x_left = lane.left_x_vals
            current_x_right = lane.right_x_vals

            left_diff = np.sum(np.absolute(prev_x_left - current_x_left))
            right_diff = np.sum(np.absolute(prev_x_right - current_x_right))

            lane_pixel_margin = 50  # How much different the new lane's x-values can be from the last lane
            diff_threshold = lane_pixel_margin*len(prev_x_left)

            if left_diff > diff_threshold or right_diff > diff_threshold:
                print(diff_threshold, int(left_diff), int(right_diff))
                print()
                good_line_diff = False
        
        # Make sure the area between the lane lines is appropriate (not too small or large)
        lane_area = np.sum(np.absolute(np.subtract(lane.right_x_vals, lane.left_x_vals)))
        area_min, area_max = 400000, 800000  # Area thesholds
        if lane_area < area_min or lane_area > area_max:
            print('Bad lane area:', lane_area)
            good_lane_area = False
        
        return (good_line_diff and good_lane_area)

    # smoothing - averages the stored lanes for a smoothing effect
    def get_avg_lane(self):
        if len(self.prev_lanes) == 0: 
            return None
        elif len(self.prev_lanes) == 1: 
            return self.prev_lanes[0]
        else:  # More than 1 previous result to average together
            n_lanes = len(self.prev_lanes)
            new_lane = self.prev_lanes[0]
            
            avg_lane = Lane()  # The averaged lane we will return (with some defaults)
            avg_lane.y_vals = new_lane.y_vals
            avg_lane.lane_lines_img = new_lane.lane_lines_img

            # Average the left and right lanes' x-values
            left_avg = new_lane.left_x_vals
            right_avg = new_lane.right_x_vals
            for i in range(1, n_lanes):
                left_avg = np.add(left_avg, self.prev_lanes[i].left_x_vals)
                right_avg = np.add(right_avg, self.prev_lanes[i].right_x_vals)

            avg_lane.left_x_vals = left_avg / n_lanes
            avg_lane.right_x_vals = right_avg / n_lanes
            
            # Average the curvatures and offsets
            avg_lane.left_curvature = sum([lane.left_curvature for lane in self.prev_lanes])/n_lanes
            avg_lane.right_curvature = sum([lane.right_curvature for lane in self.prev_lanes])/n_lanes
            avg_lane.offset = sum([lane.offset for lane in self.prev_lanes])/n_lanes

            return avg_lane

    # main functions
    def detect(self, img, show=False):
        ''' "img" can be a path or a loaded image (for the movie pipeline) '''
        # get all camera calibration images and calibrate the camera
        ret, cMat, coefs, rvects, tvects = self.calibration
            
        undist = undistort(img, cMat, coefs)
        cv2.imwrite('./undistort.jpg', undist)
        threshed = get_threshold(undist)
        cv2.imwrite('./threshold.jpg', threshed)
        warped, M, Minv = warp(threshed)
        cv2.imwrite('./warped.jpg', warped)
        lane = self.find_lane(warped)
                    
        # To speed up processing, if we've had an easy time detecting the lane, do half the updates
        if len(self.prev_lanes) == self.n_lanes_to_keep and self.n_bad_lanes == 0:
            self.n_bad_lanes += 1
            return self.draw_lane(img, self.get_avg_lane(), Minv, show=show)
        
        if self.is_good_lane(lane):  # If the lane is good (compared to previous lanes), add it to the list
            self.n_bad_lanes = 0
            self.prev_lanes.insert(0, lane)
            if len(self.prev_lanes) > self.n_lanes_to_keep: 
                self.prev_lanes.pop()
        else:
            self.n_bad_lanes += 1
            
        # If we get stuck on some bad lanes, don't reinforce it, just clear them out.
        if self.n_bad_lanes >= 12:
            print('Resetting: too many bad lanes.')
            self.n_bad_lanes = 0
            self.prev_lanes = []
            
        # If we start with some bad lanes, this will just skip the drawing
        if len(self.prev_lanes) == 0: 
            return img
                
        return self.draw_lane(img, self.get_avg_lane(), Minv, show=show)