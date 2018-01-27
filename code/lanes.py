import numpy as np
import cv2

import constants as c

###################################################################
###########   Function definitions for lane detection   ############
###################################################################

###################################################################
########################   1. Polynom fit   #######################
###################################################################

def get_lane_width(left_lane, right_lane, area_height = 100, meter_space=False):
    """
    Calculates the average lane width between left and right lane 
    in the lower area of the birdview image.
    :param left_lane: The left lane coefficients.
    :param right_lane: The right lane coefficients.
    :param area_height: Defines the pixel height of the image area
    :return: average lane width
    """
    # Determine whether to fit the line in meter or pixel space
    xmult = c.MPP_X if meter_space else 1
    # Generate x and y values for plotting
    ploty = np.linspace(c.IMG_SIZE[1] - area_height, c.IMG_SIZE[1], area_height)
    left_fitx = left_lane[0]*ploty**2 + left_lane[1]*ploty + left_lane[2]
    right_fitx = right_lane[0]*ploty**2 + right_lane[1]*ploty + right_lane[2]
    # Calculate average lane width
    average_lane_width = np.average(np.subtract(right_fitx, left_fitx)) * xmult

    return average_lane_width

def get_lateral_position(left_lane, right_lane, area_height = 100, meter_space=False):
    """
    Calculates the average distance to the lane center 
    in the lower area of the birdview image.
    :param left_lane: The left lane coefficients.
    :param right_lane: The right lane coefficients.
    :param area_height: Defines the pixel height of the image area
    :return: average lane width
    """
    # Determine whether to fit the line in meter or pixel space
    xmult = c.MPP_X if meter_space else 1
    # Generate x and y values for plotting
    ploty = np.linspace(c.IMG_SIZE[1] - area_height, c.IMG_SIZE[1], area_height)
    left_fitx = left_lane[0]*ploty**2 + left_lane[1]*ploty + left_lane[2]
    right_fitx = right_lane[0]*ploty**2 + right_lane[1]*ploty + right_lane[2]
    # Calculate distance to lane center
    dist = np.subtract(c.IMG_SIZE[0]/2, (right_fitx + left_fitx)/2) * xmult
    # Calculate average
    average_lane_width = np.average(dist)

    return average_lane_width

def get_radius(points, area_height = 100, meter_space=True):
    """
    Get the curvature radius of the given lane line at the given y coordinate.
    :param fit: The coefficients of the fitted polynomial.
    :param y_eval: The y value at which to evaluate the curvature.
    :param meter_space: Whether to calculate the fit in meter-space (True) or pixel-space (False).
    :return: The curvature radius of line at y_eval.
    """
    # Determine whether to fit the line in meter or pixel space
    ymult = c.MPP_Y if meter_space else 1
    # Fit a second-order polynom
    fit = fit_polynom(points, meter_space=meter_space)
    # Calculate the curvature radius
    ploty = np.linspace(c.IMG_SIZE[1] - area_height, c.IMG_SIZE[1], area_height)
    crv_radius = ((1 + (2*fit[0] * ploty * ymult + fit[1])**2)**1.5) / np.absolute(2*fit[0])

    return np.average(crv_radius)

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

###################################################################
######################   2. Search methods   ######################
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
    # Fit polynom to detected points
    left_fit = fit_polynom((leftx, lefty))
    right_fit = fit_polynom((rightx, righty))
    # Calculate curvature for both lanes
    left_radius = get_radius((leftx, lefty))
    right_radius = get_radius((rightx, righty))
    # Define output
    left_lane = [left_fit[0], left_fit[1], left_fit[2], left_radius]
    right_lane = [right_fit[0], right_fit[1], right_fit[2], right_radius]

    return left_lane, right_lane

def local_search(img, left_lane_hist, right_lane_hist, margin=25):
    """
    Uses a local search method if a histogram search had been already applied once.
    :param img: Binary image with birdsview on lane lines.
    :param left_lane_hist: The history of polynomial fits for the left line.
    :param right_lane_hist: The history of polynomial fits for the right line.
    :param margin: Sets the width of the windows +/- margin.
    :return: Two arrays containing the points for left and right lane respectively
    """
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Get polynom coefficients
    left_fit_prev = left_lane_hist
    right_fit_prev = right_lane_hist

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
    # Fit polynom to detected points
    left_fit = fit_polynom((leftx, lefty))
    right_fit = fit_polynom((rightx, righty))
    # Calculate curvature for both lanes
    left_radius = get_radius((leftx, lefty))
    right_radius = get_radius((rightx, righty))
    # Define output
    left_lane = [left_fit[0], left_fit[1], left_fit[2], left_radius]
    right_lane = [right_fit[0], right_fit[1], right_fit[2], right_radius]

    return left_lane, right_lane


###################################################################
#########################   3. Filtering   ########################
###################################################################


def gamma_filter(measurement, history, gamma=0.6):
    """
    Applies as simple gamma filter using the current measurement and
    the history
    :param measurement: measurement signal to be filtered
    :param history: filtered signal from previous time step
    :param gamma: weighting factor for current measurement
    :return: gamma-filtered measurement
    """
    result = measurement * gamma + (1-gamma) * history
    return result

def filter_posY(left_posY, right_posY, left_posY_hist, right_posY_hist):
    """
    Applies gamma filter to measured lateral position of left and right lane
    :param left_posY: measured left lateral position to be filtered
    :param left_posY_hist: filtered left lateral position from previous 
                            time step
    :param right_posY: measured right lateral position to be filtered
    :param right_posY_hist: filtered right lateral position from previous 
                            time step
    :return: filtered left and right lateral positon
    """
    left_posY = gamma_filter(left_head, left_posY_hist)
    right_posY = gamma_filter(right_head, right_posY_hist)

    return left_posY, right_posY

def filter_heading(left_head, right_head, left_head_hist, right_head_hist):
    """
    Applies gamma filter to measured heading angle of left and right lane
    :param left_head: measured left lane heading angle to be filtered
    :param left_head_hist: filtered left lane heading angle from previous 
                            time step
    :param right_head: measured right lane heading angle to be filtered
    :param right_head_hist: filtered right lane heading angle from previous 
                            time step
    :return: filtered left and right heading angle
    """
    left_head = gamma_filter(left_head, left_head_hist)
    right_head = gamma_filter(right_head, right_head_hist)

    return left_head, right_head

def filter_curvature(left_crv, right_crv, left_crv_hist, right_crv_hist):
    """
    Applies gamma filter to measured curvature of left and right lane
    :param left_crv: measured left lane curvature to be filtered
    :param left_crv_hist: filtered left lane curvature from previous 
                            time step
    :param right_crv: measured right lane curvature to be filtered
    :param right_crv_hist: filtered right lane curvature from previous 
                            time step
    :return: filtered left and right curvature
    """
    left_crv = gamma_filter(left_crv, left_crv_hist)
    right_crv = gamma_filter(right_crv, right_crv_hist)

    return left_crv, right_crv

def filter_radius(left_rad, right_rad, left_rad_hist, right_rad_hist):
    """
    Applies gamma filter to measured radius of left and right lane
    :param left_crv: measured left lane radius to be filtered
    :param left_crv_hist: filtered left lane radius from previous 
                            time step
    :param right_crv: measured right lane radius to be filtered
    :param right_crv_hist: filtered right lane radius from previous 
                            time step
    :return: filtered left and right radius
    """
    left_rad = gamma_filter(left_rad, left_rad_hist)
    right_rad = gamma_filter(right_rad, right_rad_hist)

    return left_rad, right_rad

def filter_lanes(left_lane, right_lane, left_lane_hist, right_lane_hist):
    """
    Applies gamma filter to all coefficients of left and right lane
    :param left_lane: left lane estimation of current time step
    :param right_lane: right lane estimation of current time step
    :param left_lane_hist: left lane estimation of previous time step
    :param right_lane_hist: right lane estimation of previous time step
    :return: filtered left and right lane
    """
    # Filter lateral position
    [left_posY, right_posY] = filter_heading(left_lane[2], right_lane[2], left_lane_hist[2], right_lane_hist[2])
    # Filter heading angle
    [left_head, right_head] = filter_heading(left_lane[1], right_lane[1], left_lane_hist[1], right_lane_hist[1])
    # Filter curvature
    [left_crv, right_crv] = filter_curvature(left_lane[0], right_lane[0], left_lane_hist[0], right_lane_hist[0])
    # Filter radius
    [left_rad, right_rad] = filter_radius(left_lane[3], right_lane[3], left_lane_hist[3], right_lane_hist[3])
    # Return filter signals
    left_lane = [left_crv, left_head, left_posY, left_rad]
    right_lane = [right_crv, right_head, right_posY, right_rad]

    return left_lane, right_lane


