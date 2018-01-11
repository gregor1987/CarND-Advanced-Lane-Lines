## Writeup for the CarND-Advanced-Lane-Lines project.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

### Camera Calibration

#### 1. This section describes how camera matrix and distortion coefficients were computed.

The code used for camera calibration in lines #17 through #112 of the file called `utils.py`) in the code folder "./code/".

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 
Input image:
![alt text](./output_images/calibration_in.jpg)

Result:
![alt text](./output_images/calibration_out.jpg)

The calibration function is called within the initialization() step in `run.py` line #47. If camera calibration parameters are available already, they will be loaded. Otherwise, the camera will be calibrated.

### Pipeline (single images)

#### 1. Undistort image: 

The following image shows the effect of undistorting the image using the calibration data received in the previous step. In the following image both, the distorted and the undistored image, have been overlayed to visualize the effect of undistortion:
![alt text](./output_images/distortion.jpg)

#### 2. Image masking

I used a combination of color/gradient thresholds and geometric masking to generate a binary image, which is implemented in lines #132 through #223 in `utils.py`).  In the following image, the effects of the different image masks can be observed. Green shows the effect of gradient masking using sobel x. Blue shows color masking. 
What can be observed in this particular plot is, why it is important to use both masking methods. Gradient thresholding shows good performance on the darker surface, but insufficient performance on the lighter surface due to the reduced contrast. Color masking compensates for that. However, color masking has issue with shadows thrown by trees for example. That is why I applied an additional gradient threshold on the color mask to reduce the effects of this downside.

![alt text](./output_images/masking.jpg)

#### 3. Perspective transform

The code for my perspective transform includes a function called `birdseye()`, which appears in lines #230 through #248 in the file `utils.py`.  This function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points as inputs. As the name suggests, it performs a perspective transform from vehicle camera perpective to birdview perspective. The function can be also used vice versa for backtransformation after lanes have been identified. The following values for source and destination points have been used:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

#### 4. Finding lanes

The masked binary image in birdview perspective builds the basis for the lane finding algorithm. The methodology to find lanes follows a 4-step process:  
    1. Check if lanes have been identified in the previous time step and make a **local search** where the lanes have been detected before  
    2. If lanes have been found with the local search approach, **check lane validity** for right and left lane respectively.  
    3. If only one lane (left or right) is invalid, **hold the lane**, which has been detected in the previous time step. If both lanes are invalid, hold both lanes detected in the previous time step.  
    4. If no lane has been detected before or if no valid lane has been detected for 5 cycles by the local search method, make a **histogram search**.  
    
**Histogram search** (`hist_search()`in lines #255 through #325 in `utils.py`)  
The histogram search approaches the lane finding problem by dividing the binary image into 9 horizontal slices. For each slice, an histogram in vertical direction is computed, which basically the sum of pixels with value 1 for each column in the image slice. This method starts with the slice on the lower part of the picture and works upwards. Thus, it can be improved by applying a geometric mask before the search. Since we know that the car is driving more or less centerd in the lane and that the lane width can be assummed to be constant, we know quite precisely where we can expect lanes to be detected in the lower part of the image. Hence, we can apply a geometric mask around that area to improve the histogram search.

![alt text](./output_images/histogram_search.jpg)  

**Local search** (`margin_search()` in lines #328 through #350 in `utils.py`)  
The local search method uses the information from the previous time step(s), if lanes had been detected before. It applies a margin around the previously detected lanes (yellow lanes in the image above) and searches only in this area locally for "hot pixels".

**Lane validity** (`check_validity()` in lines #519 through #544 in `utils.py`)  
For the lane validity check, both lanes (right/left) are checked for validity respectively. The following criteria checks have to be passed for lane validity:  
    1. **Lateral position check** (lines #442 through #478 in `utils.py`)  
       a) The lane width has to match the expected average lane width of 3.7 meters (650 pixels) with a tolerance of 50 pixels.  
       b) The right lane position shouldn't differ more than 50 pixels compared to the previous time step.  
       c) The left lane position shouldn't differ more than 50 pixels compared to the previous time step.  
    2. **Curvature check** (lines #481 through #517 in `utils.py`)  
       a) Check if both detected lanes curvatures have the same sign.  
       b) Check if the curvature of the right lane hasn't changed more than the allowed change rate.  
       c) Check if the curvature of the left lane hasn't changed more than the allowed change rate.  
  
As described before, if the validity of the detected lanes can't be confirmed, either the left, right or both lanes will be replaced with the lanes detected in the previous time step. This is done for maximum 5 consecutive cycles.  

#### 5. Radius of curvature & lane center position

`curvature()`: The calculation of the curvature radius is done in lines #378 through #391 in `utils.py`. To smoothen the output, the average of the last 5 cycles is taken.  
`get_lane_center_position()`: The calculation of the lane center position is done in lines #422 through #440 in `utils.py` and assumes the camera to be installed in the center of the car (vehicle center = image center). The current lateral position is calculated by computing the average lateral position in the lower part of the picture considering an area of 200 pixels height. From there, the distance to the image center can be computed.  

#### 6. Pipeline output 
Function: `plot_lanes()`in lines #664 through #733 in `utils.py`)  

Using the outputs from step 4 and 5 above, the lanes, radius and lateral position can be plotted on the original image. To do this, the binary image with lanes drawn on it has to be back-transformed to the original perspective. This can be easily done be using the inverted `birdview()` function (lines #230 through #248) using the source points as destination points, which have been defined in section 3. The output of this step can be seen in the following image:

![alt text](./output_images/output2.jpg)  

---

### Pipeline (video)

Here's a [link to my video result](https://youtu.be/JIvFYfw2zrY)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I had issues dealing with effects of shadow. As discussed previously, for the detection of yellow lanes on light surfaces and color mask is necessary due to low contrast. However, this leads to the fact that also other feature with lower contrast will be detected, such as shadows of trees on the surface. Here my algorithm would need some improvements with a smarte masking technique.  
Furthermore, I didn't use any filtering techniques for smoothening of the output to be able to control on these output signals. This could be achieved by means of a kalman filter or a more simpler pt1-filter (low-pass).
Generally, to make the lane detection more robust, one could think of a dynamic masking technique, which will adapt the masking on the current scene (surfaces, shadows, light, etc.).
