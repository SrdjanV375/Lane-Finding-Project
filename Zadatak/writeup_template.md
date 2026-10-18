***Lane Finding Project**

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

[image1]: ./output_images/01_undistort_chessboard.jpg "Undistorted"
[image2]: ./output_images/01_undistort_road.jpg "Road Transformed"

---

### Writeup / README

#### 1. Provide a Writeup that includes all the rubric points and how you addressed each one.

This document is the writeup. Each section answers one rubric point: where the code is, how the step
works, and an example result. Sections are completed as the corresponding step is implemented.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

**Why calibration is needed.** A camera lens bends the image: straight lines near the edges look curved. Later steps measure
the lane curvature and the car position in meters, so this distortion should be removed first.
Calibration estimates the parameters of a camera and distortion model from images of a known pattern.

**Code.** [lane_finding/calibration.py] (`calibrate_camera`, `undistort`),
run with `python calibrate.py`. The procedure follows the course exercises `09_01_calibration.py` and
`09_02_undistort.py`.

**Input.** The 20 chessboard photos in `camera_cal/`. The board has **9 x 6 inner corners** (points
where four squares meet), and its real geometry is known: the corners lie on a flat, regular grid.

**Procedure.**

1. Find the inner corners in each image (`cv2.findChessboardCorners`) and refine them to sub-pixel
   accuracy (`cv2.cornerSubPix`).
2. For each image, store the ideal corner positions on the flat board: (0,0,0), (1,0,0), ..., (8,5,0).
3. Estimate the camera matrix and distortion coefficients from all detected/ideal point pairs
   (`cv2.calibrateCamera`).
4. Project the ideal corners back with the estimated parameters (`cv2.projectPoints`) and measure
   how far they are from the detected corners (reprojection error).

Two images (`calibration7.jpg`, `calibration15.jpg`) are 1281 x 721 instead of 1280 x 720, so all images
are resized to 1280 x 720
before corner detection.

***Skipped images.** In `calibration1.jpg`, `calibration4.jpg` and `calibration5.jpg` the complete
9 x 6 grid of corners is not found, because part of the board is outside or too close to the image
border. Without the complete grid the detected corners cannot be matched to the ideal grid, so these
images are not used. **17 images** are used.

**Output.** The parameters are saved to `camera_cal/calib.npz` and loaded by the later steps:

```
Images used: 17
Reprojection error: RMS 0.846 px, mean per corner 0.686 px (09_01 formula L2/N: 0.108)
Camera matrix:
 [[1156.80    0.    665.88]
  [   0.   1151.96  388.80]
  [   0.      0.      1.  ]]
Distortion coefficients (k1, k2, p1, p2, k3): [-0.2374 -0.0879 -0.0008 -0.0001  0.1108]
```

* **Camera matrix:** `fx`, `fy` (about 1157 and 1152 px) are the focal lengths in pixels; `cx`, `cy`
  (666, 389) is the estimated optical centre, near the middle of the 1280 x 720 image.
* **Distortion coefficients:** `k1`, `k2`, `k3` describe radial distortion; the negative `k1` means
  barrel distortion (lines near the edges bow outwards). `p1`, `p2` describe tangential distortion and
  are close to zero here.
* **Reprojection error:** 0.85 px RMS and 0.69 px mean per corner. This shows a good agreement of the
  model with the detected corners on the calibration images used; it does not by itself guarantee
  accurate measurements on the road.
* The value 0.108 is the formula from the exercise. It divides the norm of all 54 corner errors by 54
  (not by the square root of 54), so it is not an error per corner in pixels.

**Removing distortion.** The function `undistort()` applies the estimated camera parameters using
`cv2.undistort(img, mtx, dist, None, mtx)`. The original camera matrix is also used as the output
camera matrix. The corrected images are then used when selecting the perspective-transform points in
the next stage.

Calibration image before and after correction (`calibration1.jpg`; it is one of the skipped images,
but the curved board edges show the effect clearly):

![alt text][image1]

**Limitation.** The calibration is applied only to 1280 x 720 images. For the 960 x 540 recordings we do
not have a confirmed matching calibration, so in this project they are not undistorted.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The same correction on a road image (`test_images/test1.jpg`). The change is visible mostly near the
edges (car hood, trees at the border); the centre of the image changes very little.

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

TODO: Add your text here!!!

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

TODO: Add your text here!!!

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

TODO: Add your text here!!!

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

TODO: Add your text here!!!

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

TODO: Add your text here!!!

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

TODO: Add your text here!!!

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

TODO: Add your text here!!!
