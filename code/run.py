import os
from os.path import join
import cv2
import numpy as np

import utils
import constants as c


def initialize():
    print('CAMERA CALIBRATION')
    [camera_mtx, dist_params] = utils.calibrate_camera()

    return camera_mtx, dist_params

def pipeline(img, history, camera_mtx, dist_params, video=False):

    left_fit_hist = history[0]
    right_fit_hist = history[1]
    crv_hist = history[2]

    print('Step 1: Undistort image')
    undistort = utils.undistort_image(img, camera_mtx, dist_params)

    print('Step 2: Apply masks')
    masked = utils.apply_masks(undistort)

    print('Step 3: Perspective transform')
    masked_topview = utils.birdseye(masked)
    # Apply geometric mask after perspective transform
    masked_topview = utils.geometric_mask(masked_topview, c.GEO1.astype(int), c.GEO2.astype(int))

    print('Step 4: Find lanes')
    [left_fit, right_fit, left_fit_hist, right_fit_hist] = utils.find_lanes(masked_topview, left_fit_hist, right_fit_hist, video)

    print('Step 5: Draw lanes & transform back')
    [output, crv_hist] = utils.plot_lanes(undistort, left_fit, right_fit, crv_hist)

    history = [left_fit_hist, right_fit_hist, crv_hist]

    return output, history


from glob import glob

def run(video=False):

    [camera_mtx, dist_params] = initialize()

    history = [[],[],[]]
    # Process single, not consecutive images
    if not video:
        file_paths = glob(join(c.IMAGES_TEST_PATH, '*.jpg'))
        file_names = os.listdir(c.IMAGES_TEST_PATH)
        imgs = utils.read_input(file_paths)
        output = []
        for img in imgs:
            result, history = pipeline(img, history, camera_mtx, dist_params)
            output.append(result)
        utils.save(output, file_names, None)
    # Process videos or consecutive images
    else:
        file_path = glob(join(c.VIDEO_TEST_PATH, '*.mp4'))
        frames = utils.read_input(file_path[0], video=True)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        output_video = cv2.VideoWriter(c.VIDEO_SAVE_PATH + 'output_video.mp4', fourcc, 25.0, (1280,720))
        # Run pipeline in batches
        for offset in range(0, len(frames), c.BATCH_SIZE):
            batch_input = frames[offset:offset+c.BATCH_SIZE]
            batch_output = []
            for frame in batch_input:
                result, history = pipeline(frame, history, camera_mtx, dist_params, video)
                batch_output.append(result)
            utils.save(batch_output, None, output_video, video)

        # Release everything if job is finished
        output_video.release()

# Run code
run(video=True)
