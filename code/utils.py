import numpy as np
import cv2
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm

from glob import glob
from os.path import join, exists, splitext

import constants as c
import lanes
import masks

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
########################   3. Masking   ###########################
###################################################################

def apply_masks(img):
    """
    Applies CLAHE normalization and LAB color masks necessary for white 
    and yello lane detection.
    :param img: input image
    :return: Combined binary with CLAHE normalization and color mask applied.
    """
    # Normalize image with CLAHE filter
    img = masks.clahe_filter(img)
    # Apply thresholds for white line detection in LAB space
    white_binary = masks.white_LAB(img)
    # Apply thresholds for yellow line detection in LAB space
    yellow_binary = masks.yellow_LAB(img)
    # Combine binaries
    combined_binary = masks.combine_binaries(white_binary,yellow_binary)

    return combined_binary


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


def find_lanes(img, history, video=False):
    """
    Finds left and right lane in a birdview image using histogram search or local search
    :param history: A tuple of arrays with the history of left and 
                    right lane coefficients from previous time step
    :return: 
        :left_lane: The left lane coefficients.
        :right_lane: The right lane coefficients.
        :history: A tuple of arrays with the history of left and 
                    right lane coefficients including the current time step.
    """
    # Decide for search method based on priority
    if video and not history == [[],[]]:
        # Load history
        left_lane_hist = history[0]
        right_lane_hist = history[1]
        # Make local search if lanes have been detected before
        print('        Local search')
        # Find lanes using local search method
        [left_lane, right_lane] = lanes.local_search(img, left_lane_hist, right_lane_hist)
        # Filter lanes
        [left_lane, right_lane] = lanes.filter_lanes(left_lane, right_lane, left_lane_hist, right_lane_hist)
        # Save history
        left_lane_hist = left_lane
        right_lane_hist = right_lane
    else:
        # Make histogram search if no lanes have been detected
        print('        Histogram search')
        # Find lanes using histogram search
        [left_lane, right_lane] = lanes.hist_search(img)
        # Save history
        left_lane_hist = left_lane
        right_lane_hist = right_lane
    # Save history
    history = [left_lane_hist, right_lane_hist]

    return left_lane, right_lane, history


###################################################################
############################   I / O  #############################
###################################################################

def read_input(file_paths, frames_max=2000, video = False):
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
        while vidcap.isOpened() and count < frames_max:
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
#######################   Plotting lanes  #########################
###################################################################

def generate_line(lane):
    """
    Generate x and y values for plotting from polynom coefficients
    :param lane: The lane coefficients
    :return: The points of the lane (y, x) in y and x position
    """
    # Generate x and y values for plotting
    ploty = np.linspace(0, c.IMG_SIZE[1]-1, c.IMG_SIZE[1] )
    fitx = lane[0]*ploty**2 + lane[1]*ploty + lane[2]

    return (fitx, ploty)

def plot_lanes(img, left_lane, right_lane):
    """
    Draws left and right lane on the input image in birdview and
    transforms it back to the original perspective.
    :param img: Input image in birdview
    :param left_lane: The left lane coefficients.
    :param right_lane: The right lane coefficients.
    :return: 
            output_img: image with drawn detected lane.
    """
    output_img = np.empty_like(img)

    # Create image to draw lines on
    lane_img = np.zeros_like(img).astype(np.uint8)
    line_img = np.zeros_like(img).astype(np.uint8)
    # Generate x and y values for plotting
    left_line = generate_line(left_lane)
    right_line = generate_line(right_lane)
    # Recast
    left_pts = np.array([np.transpose(np.vstack(left_line))])
    right_pts = np.array([np.flipud(np.transpose(np.vstack(right_line)))])
    pts = np.hstack((left_pts, right_pts))

    t_l = 20
    # Recast
    left_pts = np.array([np.transpose(np.vstack((left_line[0]-t_l,left_line[1])))])
    right_pts = np.array([np.flipud(np.transpose(np.vstack((left_line[0]+t_l,left_line[1]))))])
    pts_left = np.hstack((left_pts, right_pts))

    # Recast
    left_pts = np.array([np.transpose(np.vstack((right_line[0]-t_l,right_line[1])))])
    right_pts = np.array([np.flipud(np.transpose(np.vstack((right_line[0]+t_l,right_line[1]))))])
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
    # Calculate distance to lane center
    dist = lanes.get_lateral_position(left_lane, right_lane, meter_space=True)
    # Calculate curvature
    curvature = (left_lane[3] + right_lane[3])/2
    # Calculate lane width
    lane_width = lanes.get_lane_width(left_lane, right_lane, meter_space=True)

    # Plot curvature radius, center distance and lane width on output image
    text_color = (255, 255, 255)
    cv2.putText(output_img, "Curvature Radius: " + '{:.2f}'.format(curvature/1000) + 'km', (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
    cv2.putText(output_img, "Distance from Center: " + '{:.2f}'.format(dist) + 'm', (20, 80), 
        cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
    cv2.putText(output_img, "Measured lane width: " + '{:.2f}'.format(lane_width) + 'm', (20, 110), 
        cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

    return output_img
