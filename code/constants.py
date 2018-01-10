from os import makedirs
from os.path import join, exists
import numpy as np
import matplotlib.image as mpimg


def get_dir(directory):
    """
    Creates the given directory if it does not exist.
    @param directory: The path to the directory.
    @return: The path to the directory.
    """
    if not exists(directory):
        makedirs(directory)
    return directory

def getSourcePts(img_size):
    src = np.float32(
     [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
     [((img_size[0] / 6) - 10), img_size[1]],
     [(img_size[0] * 5 / 6) + 60, img_size[1]],
     [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
    return src   
    
def getDestinationPts(img_size):
    dst = np.float32(
     [[(img_size[0] / 4), 0],
     [(img_size[0] / 4), img_size[1]],
     [(img_size[0] * 3 / 4), img_size[1]],
     [(img_size[0] * 3 / 4), 0]])
    return dst

DATA_PATH = '../data/'
CALIBRATION_PATH = join(DATA_PATH, 'camera_cal/')
CALIBRATION_DATA = join(CALIBRATION_PATH, 'calibration_data.p')
TEST_PATH = join(DATA_PATH, 'test_images/')

SAVE_PATH = join(DATA_PATH, 'output_images/')

# Define image size
IMG_SIZE = (1280, 720)

# Points picked from an image with straight lane lines.
SRC = getSourcePts(IMG_SIZE)

# Mapping from those points to a rectangle for a birdseye view.
DST = getDestinationPts(IMG_SIZE)

# Define points for the geometric mask
GEO1 = np.float32(
     [[50, 0],
     [(IMG_SIZE[0] / 6), IMG_SIZE[1]],
     [(IMG_SIZE[0] * 3 / 10), IMG_SIZE[1]],
     [(IMG_SIZE[0] * 4 / 10), 0]])

GEO2 = np.float32(
     [[(IMG_SIZE[0] * 6 / 10), 0],
     [(IMG_SIZE[0] * 7 / 10), IMG_SIZE[1]],
     [(IMG_SIZE[0] * 5 / 6), IMG_SIZE[1]],
     [(IMG_SIZE[0] - 50), 0]])

# The maximum number of previous frames for which lane fits will be saved
HIST_LENGTH = 7
CRV_HIST = 7

# Average lane width in pixels
AVG_LN_WIDTH = 650

# Batch size for video processing
BATCH_SIZE = 100

# meters per pixel in y dimension
MPP_Y = 30. / 720
# meters per pixel in x dimension
MPP_X = 3.7 / AVG_LN_WIDTH

# Camera calibration
NX = 9 # number of inside corners in x
NY = 5 # number of inside corners in y
