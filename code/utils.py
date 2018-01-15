import numpy as np
import cv2
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from glob import glob
from os.path import join, exists, splitext

import constants as c

###################################################################
###################   1. Camera calibration   #####################
###################################################################

def calibrate_camera():
    """
    This function checks if calibration data is already available in c.CALIBRATION_DATA. 
    If not, the camera is calibrated using images from c.CALIBRATION_PATH.
    :return: camera matrix 'camera_mtx' and distortion parameters 'dist_params'
    """
    # Check if camera has been calibrated previously.
    if exists(c.CALIBRATION_DATA):
        # Return pickled calibration data.
        pickle_dict = pickle.load(open(c.CALIBRATION_DATA, "rb"))
        camera_mtx = pickle_dict["camera_mtx"]
        dist_params = pickle_dict["dist_params"]

        print('        Calibration data loaded')

        return camera_mtx, dist_params
    
    # If no camera calibration data exists, calibrate camera
    print('        Calibrating camera...')

    # For every calibration image, get object points and image points by finding chessboard corners.
    obj_points = []  # 3D points in real world space.
    img_points = []  # 2D points in image space.

    # Prepare constant object points, like (0,0,0), (1,0,0), (2,0,0) ....,(9,6,0).
    obj_points_const = np.zeros((c.NY * c.NX, 3), np.float32)
    obj_points_const[:, :2] = np.mgrid[0:c.NX, 0:c.NY].T.reshape(-1, 2)

    print('        Load calibration images')
    file_paths = glob(join(c.CALIBRATION_PATH, '*.jpg'))

    for path in file_paths:
        # Read in single calibration image
        img = mpimg.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (c.NX, c.NY), None)

        if ret:
            obj_points.append(obj_points_const)
            img_points.append(corners)

    # Calculate camera matrix and distortion parameters and return.
    ret, camera_mtx, dist_params, _, _ = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None)

    assert ret, 'CALIBRATION FAILED'  # Make sure calibration didn't fail.

    # Save calibration data
    pickle_dict = {'camera_mtx': camera_mtx, 'dist_params': dist_params}
    pickle.dump(pickle_dict, open(c.CALIBRATION_DATA, 'wb'))

    print('        Camera calibrated')

    return camera_mtx, dist_params

def calibration_src(img):
    """
    Calculates source points for calibration images
    :param img: input image  used for calibration
    :return: source points for calibration images
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (c.NX, c.NY), None)
    if ret:
        x = []
        y = []
        for corner in corners:
            x.append(corner[0][0])
            y.append(corner[0][1])
    
    assert ret, 'NO CORNERS FOUND'

    return np.float32([[x[c.NX-1],y[c.NX-1]],[x[len(x)-1],y[len(x)-1]],[x[len(x)-c.NX],y[len(x)-c.NX]],[x[0],y[0]]])
    
def calibration_dst():
    """
    Calculates destination points for calibration images
    :return: destination points for calibration images
    """
    x_sqr_size = c.IMG_SIZE[0]/(c.NX + 1)
    y_sqr_size = c.IMG_SIZE[1]/(c.NY + 1)
    return np.float32([[c.IMG_SIZE[0]-x_sqr_size,y_sqr_size],[c.IMG_SIZE[0]-x_sqr_size,c.IMG_SIZE[1]-y_sqr_size],[x_sqr_size, c.IMG_SIZE[1]-y_sqr_size],[x_sqr_size,y_sqr_size]])   

def perspective_transform(img, src, dst):
    """
    Performs perspective transform from source points to destination points
    :param img: input image
    :param src: array of source points
    :param dst: array of destination points
    :return: Warped image
    """
    # Get transformation matrix M
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp perspective
    return cv2.warpPerspective(img, M, c.IMG_SIZE, flags=cv2.INTER_LINEAR)

###################################################################
##################   2. Distortion correction   ###################
###################################################################

def undistort_image(img, camera_mtx, dist_params):
    """
    Undistorts image using camera calibration parameters
    :param img: input image
    :param camera_mtx: camera matrix
    :param dist_params: camera distortion parameters
    :return: Undistorted image 
    """
    return cv2.undistort(img, camera_mtx, dist_params, None, camera_mtx)

###################################################################
##########################   3. Masks   ###########################
###################################################################

def color_mask(img, s_thresh=(150, 255), sx_thresh=(35, 100)):
    """
    Converts input image from RGB to HLS space and applies color mask to s-channel of input image.
    Furthermore an additional sobel x mask is applied on color masked image.
    :param img: input image
    :param s_thresh: sets the threshold parameters for the color mask.
    :param sx_thresh: sets the threshold parameters for sobel x.
    :return: Binary image with pixels within thresholds set to 1. 
    """
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hls[:,:,2]
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Sobel x
    sobelx = cv2.Sobel(s_binary, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    # Threshold x gradient
    s_binary = np.zeros_like(scaled_sobel)
    s_binary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    return s_binary

def sobel_mask(img, sx_thresh=(35, 100)):
    """
    Converts input image from RGB to HLS space and applies sobel x mask to l-channel of input image.
    :param img: input image
    :param sx_thresh: sets the threshold parameters for sobel x.
    :return: Binary image with pixels within thresholds set to 1. 
    """
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    return sxbinary

def apply_masks(img):
    """
    Applies all masks (color mask and sobel mask) necessary for lane detection.
    :param img: input image
    :return: Combined binary with color mask and sobel masked applied.
    """
    # Threshold x gradient
    sxbinary = sobel_mask(img) 
    # Threshold color channel
    s_binary = color_mask(img)
    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), (sxbinary == 1), (s_binary == 1))) * 255
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    # combined_binary[(sxbinary == 1)] = 1
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary

def geometric_mask(img, vertices1, vertices2):
    """
    Applies a geometric image mask.
    :param img: input image
    :param vertices1: vertices of region of interest 1
    :param vertices2: vertices of region of interest 2
    :return: Only keeps the region of the image defined by the polygon
             formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, [vertices1], ignore_mask_color)
    cv2.fillPoly(mask, [vertices2], ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


###################################################################
#################   4. Perspective transform   ####################
###################################################################

def birdseye(img, inverse=False):
    """
    Performs a perspective transform from/to birdview perspective.
    :param img: input image
    :param inverse: if TRUE, perspective transform from vehicle ego view to birdview
                    if FALSE, perspective transform from birdview to vehicle ego view
    :return: birdsview image or vehicle ego view image
    """
    # Define source and destination points
    if inverse:
        src = c.DST
        dst = c.SRC
    else:
        src = c.SRC
        dst = c.DST
    # Get transformation matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Perform perspective transform
    return cv2.warpPerspective(img, M, c.IMG_SIZE, flags=cv2.INTER_LINEAR)

###################################################################
######################   5. Finding lanes   #######################
###################################################################


def hist_search(img, margin=100, nwindows=9):
    """
    Uses a sliding histogram to search for lane lines in a binary birdseye image.
    :param img: Binary image with birdsview on lane lines.
    :param margin: Sets the width of the windows +/- margin.
    :param nwindows: Sets the number of sliding windows
    :return: Two arrays containing the points for left and right lane respectively
    """
    # Assuming you have created a warped binary image called "img"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((img, img, img))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    return (leftx, lefty), (rightx, righty)


def margin_search(img, left_fit_hist, right_fit_hist, margin=25):
    """
    Uses a local search method if a histogram search had been already applied once.
    :param img: Binary image with birdsview on lane lines.
    :param left_fit_hist: The history of polynomial fits for the left line.
    :param right_fit_hist: The history of polynomial fits for the right line.
    :param margin: Sets the width of the windows +/- margin.
    :return: Two arrays containing the points for left and right lane respectively
    """
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Must exist because of check in find_lines()
    left_fit_prev = [elt for elt in left_fit_hist[-c.HIST_LENGTH:] if elt is not None][-1]
    right_fit_prev = [elt for elt in right_fit_hist[-c.HIST_LENGTH:] if elt is not None][-1]

    left_lane_inds = ((nonzerox > (left_fit_prev[0]*(nonzeroy**2) + left_fit_prev[1]*nonzeroy + 
    left_fit_prev[2] - margin)) & (nonzerox < (left_fit_prev[0]*(nonzeroy**2) + 
    left_fit_prev[1]*nonzeroy + left_fit_prev[2] + margin))) 

    right_lane_inds = ((nonzerox > (right_fit_prev[0]*(nonzeroy**2) + right_fit_prev[1]*nonzeroy + 
    right_fit_prev[2] - margin)) & (nonzerox < (right_fit_prev[0]*(nonzeroy**2) + 
    right_fit_prev[1]*nonzeroy + right_fit_prev[2] + margin)))

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    return (leftx, lefty), (rightx, righty)

def fit_polynom(points, meter_space=False, order=2):
    """
    Fits a second-order polynomial to the given points in image and returns polynom coefficients
    :param points: The points on which to fit the line.
    :param meter_space: Whether to calculate the fit in meter-space (True) or pixel-space (False).
    :return: The coefficients of the fitted polynomial.
    """
    # Determine whether to fit the line in meter or pixel space
    ymult = c.MPP_Y if meter_space else 1
    xmult = c.MPP_X if meter_space else 1

    # Fit a second-order polynom
    fit = np.polyfit(points[1] * ymult, points[0] * xmult, order)

    return fit

def curvature(fit, y_eval=c.IMG_SIZE[1]-200, meter_space=False):
    """
    Get the curvature radius of the given lane line at the given y coordinate.
    :param fit: The coefficients of the fitted polynomial.
    :param y_eval: The y value at which to evaluate the curvature.
    :param meter_space: Whether to calculate the fit in meter-space (True) or pixel-space (False).
    :return: The curvature radius of line at y_eval.
    """
    # Determine whether to fit the line in meter or pixel space
    ymult = c.MPP_Y if meter_space else 1

    curverad = ((1 + (2*fit[0]*y_eval*ymult + fit[1])**2)**1.5) / np.absolute(2*fit[0])

    return curverad

def generate_line(fit):
    """
    Generate x and y values for plotting from polynom coefficients
    :param fit: Polynom coefficients from cv2.polyfit
    :return: The points of the lane (y, x) in y and x position
    """
    # Generate x and y values for plotting
    ploty = np.linspace(0, c.IMG_SIZE[1]-1, c.IMG_SIZE[1] )
    fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]

    return (fitx, ploty)

def get_avg_lane_width(left_fit, right_fit, area_height = 200):
    """
    Calculates the average lane width of the fitted lines in the lower area of the birdview image
    :param left_fit: The polynomial fit for the left line.
    :param right_fit: The polynomial fit for the right line.
    :param area_height: Defines the pixel height of the image area
    :return: average lane width
    """
    # Generate x and y values for plotting
    ploty = np.linspace(c.IMG_SIZE[1] - area_height, c.IMG_SIZE[1], area_height)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    avg_lane_width = np.average(np.subtract(right_fitx, left_fitx))

    return avg_lane_width

def get_lane_center_position(left_fit, right_fit, area_height = 200, meter_space=False):
    """
    Returns the lateral distance to the lane center
    :param left_fit: The polynomial fit for the left line.
    :param right_fit: The polynomial fit for the right line.
    :return: Lateral distance to lane center in meter
    """
    # Determine whether to fit the line in meter or pixel space
    xmult = c.MPP_X if meter_space else 1
    # Generate x and y values for plotting
    ploty = np.linspace(c.IMG_SIZE[1] - area_height, c.IMG_SIZE[1], area_height)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    avg_pos = np.average(np.add(right_fitx, left_fitx)/2)

    avg_dist_to_lanecenter = (avg_pos - c.IMG_SIZE[0]/2) * xmult

    return avg_dist_to_lanecenter

def check_lateral_position(left_fit, right_fit, left_fit_hist, right_fit_hist):
    """
    Checks the lateral position of the detected left and right lane for validity
    :param left_fit: The polynomial fit for the left line.
    :param right_fit: The polynomial fit for the right line.
    :param left_fit_hist: The history of polynomial fits for the left line.
    :param right_fit_hist: The history of polynomial fits for the right line.
    :return: Two boolean, one for each lane indicating TRUE if the lane lateral position is valid
    """
    # Get fit from previous time step for left and right line
    prev_fit_left = [elt for elt in left_fit_hist[-c.HIST_LENGTH:] if elt is not None][-1]
    prev_fit_right = [elt for elt in right_fit_hist[-c.HIST_LENGTH:] if elt is not None][-1]

    # Define limits for lane width
    avg_lane_width = c.AVG_LN_WIDTH
    tolerance = 50

    # Define lateral position qualifier
    left_pos_valid = False
    right_pos_valid = False

    # Check if lane width is valid
    if avg_lane_width - tolerance < get_avg_lane_width(left_fit, right_fit) < avg_lane_width + tolerance:
        left_pos_valid = True
        right_pos_valid = True
    else:
        print('        Lane width invalid')
        # Check right lateral position is valid
        if (np.absolute(right_fit[2] - prev_fit_right[2]) > tolerance):
            print('        Right lateral position invalid')
            right_pos_valid = False
        # Check left lateral position is valid
        if (np.absolute(left_fit[2] - prev_fit_left[2]) > tolerance):
            print('        Left lateral position invalid')
            left_pos_valid = False

    return left_pos_valid, right_pos_valid


def check_curvature(left_fit, right_fit, left_fit_hist, right_fit_hist, valid_change = 1.0):
    """
    Checks the curvature of detected left and right lane for validity
    :param left_fit: The polynomial fit for the left line.
    :param right_fit: The polynomial fit for the right line.
    :param left_fit_hist: The history of polynomial fits for the left line.
    :param right_fit_hist: The history of polynomial fits for the right line.
    :param valid_change: sets the maximum allowed difference (in %) to last cycle detection
    :return: Two boolean, one for each lane indicating TRUE if the lane curvature is valid
    """
    # Get fit from previous time step for left and right line
    prev_fit_left = [elt for elt in left_fit_hist[-c.HIST_LENGTH:] if elt is not None][-1]
    prev_fit_right = [elt for elt in right_fit_hist[-c.HIST_LENGTH:] if elt is not None][-1]
    # Define curvature qualifier
    # curvature_valid = False
    left_crv_valid = False
    right_crv_valid = False
    # 2.1. Check if right and left line have the same curvature sign
    if np.sign(left_fit[0]) == np.sign(right_fit[0]):
        left_crv_valid = True
        right_crv_valid = True
        
        # Invalidate left lane if the curvature changed more than 70 %
        if (np.absolute(left_fit[0]-prev_fit_left[0])/np.absolute(prev_fit_left[0]) > valid_change):
            print('        Left curvature invalid')
            left_crv_valid = False
            # left_fit = prev_fit_left

        # Invalidate right lane if the curvature changed more than 70 %
        if (np.absolute(right_fit[0]-prev_fit_right[0])/np.absolute(prev_fit_right[0]) > valid_change):
            print('        Right curvature invalid')
            right_crv_valid = False
    else:
        print('        Different curvature signs')


    return left_crv_valid, right_crv_valid

def check_validity(left_fit, right_fit, left_fit_hist, right_fit_hist):
    """
    Checks the overall validity of the detected left and right lane
    :param left_fit: The polynomial fit for the left line.
    :param right_fit: The polynomial fit for the right line.
    :param left_fit_hist: The history of polynomial fits for the left line.
    :param right_fit_hist: The history of polynomial fits for the right line.
    :return: Two boolean, one for each lane indicating TRUE if the lane is valid
    """
    left_valid = False
    right_valid = False
    # Check lateral position validity
    [left_pos_valid, right_pos_valid] = check_lateral_position(left_fit, right_fit, left_fit_hist, right_fit_hist)
    # Check curvature validity
    [left_crv_valid, right_crv_valid] = check_curvature(left_fit, right_fit, left_fit_hist, right_fit_hist)
    # If more than one coefficient invalid, invalidate lane
    if sum([left_crv_valid, right_crv_valid, left_pos_valid, right_pos_valid]) == 4:
        left_valid = True
        right_valid = True
    elif sum([left_crv_valid, right_crv_valid, left_pos_valid, right_pos_valid]) < 4:
        if left_crv_valid and left_pos_valid:
           left_valid = True
        if right_crv_valid and right_pos_valid:
           right_valid = True

    return left_valid, right_valid

def find_lanes(img, left_fit_hist, right_fit_hist, video=False):
    """
    Finds left and right lane in a birdview image using histogram search or local search
    :param left_fit_hist: The history of polynomial fits for the left line.
    :param right_fit_hist: The history of polynomial fits for the right line.
    :return: 
        :left_fit: The polynomial fit for the left line for this cycle
        :right_fit: The polynomial fit for the right line for this cycle
        :left_fit_hist: The history of polynomial fits for the left line (incl. the current cycle if valid).
        :right_fit_hist: The history of polynomial fits for the right line (incl. the current cycle if valid).
    """
    if ((len([elt for elt in left_fit_hist[-c.HIST_LENGTH:] if elt is not None]) != 0) and 
    (len([elt for elt in right_fit_hist[-c.HIST_LENGTH:] if elt is not None]) != 0) and
    video):
        # Prio 1: Make local search
        print('        Local search')
        [left_lane_pts, right_lane_pts] = margin_search(img, left_fit_hist, right_fit_hist)
        left_fit = fit_polynom(left_lane_pts)
        right_fit = fit_polynom(right_lane_pts)

        [left_valid, right_valid] = check_validity(left_fit, right_fit, left_fit_hist, right_fit_hist)
        if left_valid and right_valid:
            left_fit_hist.append(left_fit)
            right_fit_hist.append(right_fit)
        # Prio 2: Replace single lane with lane from previous time step if only one lane is invalid
        else:
            if left_valid and right_valid == False:
                right_fit = [elt for elt in right_fit_hist[-c.HIST_LENGTH:] if elt is not None][-1]
                print('        Right lane hold')
                left_fit_hist.append(left_fit)
                right_fit_hist.append(None)
            elif right_valid and left_valid == False:
                left_fit = [elt for elt in left_fit_hist[-c.HIST_LENGTH:] if elt is not None][-1]
                print('        Left lane hold')
                left_fit_hist.append(None)
                right_fit_hist.append(right_fit)
        # Prio 3: Replace both lanes with lanes from previous time step if both are invalid
            else:
                left_fit = [elt for elt in left_fit_hist[-c.HIST_LENGTH:] if elt is not None][-1]
                right_fit = [elt for elt in right_fit_hist[-c.HIST_LENGTH:] if elt is not None][-1]
                print('        Both lanes hold')
                left_fit_hist.append(None)
                right_fit_hist.append(None)

    else:
        # Prio 4: Make histogram search if no lanes had been detected before or history is too old
        print('        Histogram search')
        [left_lane_pts, right_lane_pts] = hist_search(img)
        left_fit = fit_polynom(left_lane_pts)
        right_fit = fit_polynom(right_lane_pts)
        left_fit_hist.append(left_fit)
        right_fit_hist.append(right_fit)

    return left_fit, right_fit, left_fit_hist, right_fit_hist


###################################################################
############################   I / O  #############################
###################################################################

def read_input(file_paths, frames=2000, video = False):
    """
    Reads images from input file paths into a numpy array. Paths can either be .jpg for single images
    or .mp4 for videos.
    :param file_paths: Array containing the file paths.
    :param video: if set to TRUE, an input video will be processed.
    :param frames: Sets the maximum number of frames which shall be read in
    :return: A numpy array of images.
    """
    if not video:
        frames = []
        for path in file_paths:
            #  Read in single image.
            img = mpimg.imread(path)
            frames.append(img)
    else:
        # Input is a video.
        vidcap = cv2.VideoCapture(file_paths)
        length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Load frames
        frames_list = []
        count = 1
        while vidcap.isOpened() and count < frames:
            ret, frame = vidcap.read()
            print('Reading frame',count)
            if ret:
                frames_list.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                count += 1
            else:
                break

        vidcap.release()

        frames = np.array(frames_list)

    return frames


def save(imgs, file_names, video_out, video = False):
    """
    Saves imgs to file using original file_names.
    :param imgs: The frames to save. A single image for .jpgs, or multiple frames for .mp4s.
    :param file_names: array containing file names of input imgs for naming of output images.
    :param video_out: cv2.VideoWriter object for video writing.
    :param video: if set to TRUE, an input video will be processed.
    """
    if not video:
        for i in range(len(imgs)):
            cv2.imwrite(c.IMAGES_SAVE_PATH + 'output_' + file_names[i], cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB))
    else:
        print(video_out.isOpened())
        for i in range(len(imgs)):
            print('Writing frame '+str(i))
            video_out.write(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB))


###################################################################
########################   Other Helpers  #########################
###################################################################


def plot_lanes(img, left_fit, right_fit, crv_hist):
    """
    Draws lanes on the input image give the fit parameters 'left_fit'/'right_fit' 
    for left and right lane.
    :param img: Input image
    :param left_fit: second-order fit parameters for left lane.
    :param right_fit: second-order fit parameters for left lane.
    :param crv_hist: array containing history of curvature estimations.
    :return: 
            output_img: image with drawn detected lane.
            crv_hist: array containing history of curvature estimations.
    """
    output_img = np.empty_like(img)

    # Create image to draw lines on
    lane_img = np.zeros_like(img).astype(np.uint8)
    line_img = np.zeros_like(img).astype(np.uint8)
    # Generate x and y values for plotting
    left_lane = generate_line(left_fit)
    right_lane = generate_line(right_fit)

    # Recast
    left_pts = np.array([np.transpose(np.vstack(left_lane))])
    right_pts = np.array([np.flipud(np.transpose(np.vstack(right_lane)))])
    pts = np.hstack((left_pts, right_pts))

    t_l = 20
    # Recast
    left_pts = np.array([np.transpose(np.vstack((left_lane[0]-t_l,left_lane[1])))])
    right_pts = np.array([np.flipud(np.transpose(np.vstack((left_lane[0]+t_l,left_lane[1]))))])
    pts_left = np.hstack((left_pts, right_pts))

    # Recast
    left_pts = np.array([np.transpose(np.vstack((right_lane[0]-t_l,right_lane[1])))])
    right_pts = np.array([np.flipud(np.transpose(np.vstack((right_lane[0]+t_l,right_lane[1]))))])
    pts_right = np.hstack((left_pts, right_pts))

    # Draw the lane onto lane image and invert perspective transform
    cv2.fillPoly(line_img, np.int_([pts_right]), (204, 51, 51))
    cv2.fillPoly(line_img, np.int_([pts_left]), (204, 51, 51))
    cv2.fillPoly(lane_img, np.int_([pts]), (51, 166, 204))
    
    lane_overlay = birdseye(lane_img, inverse=True)
    line_overlay = birdseye(line_img, inverse=True)

    # Combine the result with the original image
    output_img = cv2.addWeighted(img, 1, lane_overlay, 0.3, 0)
    output_img = cv2.addWeighted(output_img, 1, line_overlay, 1, 0)

    # Calculate average curvature
    left_curvature = curvature(left_fit, meter_space=True)
    right_curvature = curvature(right_fit, meter_space=True)
    crv_hist.append(left_curvature)
    # crv_hist.append((left_curvature + right_curvature) / 2)
    if (len(crv_hist) < c.CRV_HIST):
        crv = np.average(crv_hist)
    else:
        crv = np.average(crv_hist[-c.CRV_HIST:])

    # Calculate distance to lane center
    dist = get_lane_center_position(left_fit, right_fit, meter_space=True)

    # Plot curvature and distance to lane center on output image
    text_color = (255, 255, 255)
    cv2.putText(output_img, "Curvature Radius: " + '{:.2f}'.format(crv/1000) + 'km', (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
    cv2.putText(output_img, "Distance from Center: " + '{:.2f}'.format(dist) + 'm', (20, 80), 
        cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

    return output_img, crv_hist
