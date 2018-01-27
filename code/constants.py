from os.path import join
import numpy as np

###################################################################
#####################   Define constants   ########################
###################################################################

DATA_PATH = '../data/'
CALIBRATION_PATH = join(DATA_PATH, 'camera_cal/')
CALIBRATION_DATA = join(CALIBRATION_PATH, 'calibration_data.p')
IMAGES_TEST_PATH = join(DATA_PATH, 'test_images/')
FRAMES_TEST_PATH = join(DATA_PATH, 'test_frames/')
IMAGES_SAVE_PATH = join(DATA_PATH, 'output_images/')
VIDEO_TEST_PATH = join(DATA_PATH, 'test_video/')
VIDEO_SAVE_PATH = join(DATA_PATH, 'output_video/')

# Define image size
IMG_SIZE = (1280, 720)

# Define source and destination points points for the geometric mask
SRC = np.float32(
     [[580, 460],
     [260, 691],
     [1067, 691],
     [705, 460]])

DST = np.float32(
     [[(IMG_SIZE[0] / 4), 0],
     [(IMG_SIZE[0] / 4), IMG_SIZE[1]],
     [(IMG_SIZE[0] * 3 / 4), IMG_SIZE[1]],
     [(IMG_SIZE[0] * 3 / 4), 0]])

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

# Average lane width in pixels
AVG_LN_WIDTH = 640

# Batch size for video processing
BATCH_SIZE = 100

# meters per pixel in y dimension
MPP_Y = 30. / 720
# meters per pixel in x dimension
MPP_X = 3.7 / AVG_LN_WIDTH

# Camera calibration
NX = 9 # number of inside corners in x
NY = 5 # number of inside corners in y
