import numpy as np
import cv2

###################################################################
##################   Mask definition & helpers   ##################
###################################################################

def combine_binaries(binary_1, binary_2):
    """
    Combines two binaries into one combined binary (logical OR)
    :param binary_1: first input binary
    :param binary_2: second input binary 
    :return: combined binary
    """
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(binary_1)
    # combined_binary[(sxbinary == 1)] = 1
    combined_binary[(binary_1 == 1) | (binary_2 == 1)] = 1

    return combined_binary

def clahe_filter(img):
    """
    Applies CLAHE filter (Contrast Limited Adaptive Histogram Equalization)
    in LAB color space to improve contrast for varying lighting conditions
    :return: CLAHE normalized image 
    """
    img_temp = np.copy(img)
    img_temp = cv2.cvtColor(img_temp, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(1,1))
    img_temp[:,:,0] = clahe.apply(img_temp[:,:,0])
    result = cv2.cvtColor(img_temp, cv2.COLOR_LAB2RGB)

    return result

def channel_thresh(img, channel, thresh=(0,255)):
    """
    Takes in an image and returns thresholded binary image for specified channel
    :param img: input image
    :param channel: channel of image which shall be thresholded
    :return: thresholded binary image
    """
    channel = img[:,:,channel]
    binary_output = np.zeros_like(channel)
    binary_output[(channel > thresh[0]) & (channel <= thresh[1])] = 1

    return binary_output

def yellow_LAB(img, l_thresh=(130, 255), a_thresh=(100, 150), b_thresh=(145, 210)):
    """
    Converts input image from RGB to LAB space, applies thresholds
    to all three channels and returns the combined binary (logical AND).
    The threshold are tuned to detect yellow lines.
    :param img: input image
    :param l_thresh: sets the threshold parameters for the l-channel.
    :param a_thresh: sets the threshold parameters for the a-channel.
    :param b_thresh: sets the threshold parameters for the b-channel.
    :return: Binary image with pixels within thresholds set to 1. 
    """
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    LAB = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float)
    # Return thresholed LAB channels
    l_binary = channel_thresh(LAB, 0, thresh=l_thresh)
    a_binary = channel_thresh(LAB, 1, thresh=a_thresh)
    b_binary = channel_thresh(LAB, 2, thresh=b_thresh)
    # Combine binaries
    binary_yellow = np.zeros_like(l_binary)
    binary_yellow[(l_binary == 1) & (a_binary == 1) & (b_binary == 1)] = 1

    return binary_yellow

def white_LAB(img, l_thresh=(230, 255), a_thresh=(120, 140), b_thresh=(120, 140)):
    """
    Converts input image from RGB to LAB space, applies thresholds
    to all three channels and returns the combined binary (logical AND).
    The threshold are tuned to detect white lines.
    :param img: input image
    :param l_thresh: sets the threshold parameters for the l-channel.
    :param a_thresh: sets the threshold parameters for the a-channel.
    :param b_thresh: sets the threshold parameters for the b-channel.
    :return: Binary image with pixels within thresholds set to 1. 
    """
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    LAB = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float)
    # Return thresholed LAB channels
    l_binary = channel_thresh(LAB, 0, thresh=l_thresh)
    a_binary = channel_thresh(LAB, 1, thresh=a_thresh)
    b_binary = channel_thresh(LAB, 2, thresh=b_thresh)
    # Combine binaries
    binary_white = np.zeros_like(l_binary)
    binary_white[(l_binary == 1) & (a_binary == 1) & (b_binary == 1)] = 1

    return binary_white

