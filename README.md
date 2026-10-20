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

[image1]: ./docs/01_undistort_chessboard.jpg "Undistorted"
[image2]: ./docs/01_undistort_road.jpg "Road Transformed"
[image3]: ./docs/02_binary.jpg "Binary Example"
[image3b]: ./docs/02_binary_test5.jpg "Binary test5"
[image4]: ./docs/03_perspective.jpg "Warp Example"
[image4b]: ./docs/03_binary_warped.jpg "Binary in bird's-eye view"
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


**Goal.** Make a black-and-white image where only the lane markings are white. The next steps look
for the lines only among these white pixels.

**Code.** [lane_finding/thresholds.py](Zadatak/lane_finding/thresholds.py) (`lane_masks`), run with
`python binary_step.py`. The input is the undistorted image from step 1, slightly blurred
(`GaussianBlur` 5 x 5) to remove noise.

The binary image is made of two masks, one for **yellow** and one for **white** lines:

* **Yellow** - I first tried HSV with `cv2.inRange`, but it also picked up dry grass (2nd column
  below). So I used the **b channel of Lab** instead: b is high for yellow and does not depend much
  on brightness, so it works both in shadow and in sun. A pixel is yellow if its b value is high and
  higher than its surroundings.
* **White** - the pixel must be:
  1. **bright** in the R channel. The threshold is taken from the histogram of the road in front of
     the car, not fixed: on dark asphalt it is 126, on bright concrete 220. A fixed threshold made the
     whole concrete road white;
  2. **narrow** - morphological top-hat keeps only thin bright stripes, so big bright areas (concrete
     patches, cars) are removed;
  3. **not coloured** (low HSV saturation) - otherwise brown dirt next to the road was taken as a line.

At the end, everything wider than a lane marking is removed.

I also tried the **Sobel x** gradient. It finds the lines, but also every shadow, crack and tree (last
column), so I did not use it.

Columns: image, HSV yellow (first try), Lab yellow, white mask, **final binary**, Sobel x:

![alt text][image3]

Final result on `test5.jpg` (concrete with shadows):

![alt text][image3b]

Some white pixels remain in the sky and trees. They are removed in the next step with a region of
interest (only the road in front of the car is kept).


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

**Goal.** In the camera image the lane lines come together towards the horizon. Seen from above
(bird's-eye view) they are parallel, so it is easier to find them and fit a curve.

**Code.** [lane_finding/perspective.py](Zadatak/lane_finding/perspective.py) (class `Perspective`), run with
`python perspective_step.py`. It uses `cv2.getPerspectiveTransform` and `cv2.warpPerspective`, like the
`warper()` function given with the assignment.

**How I chose the points.** On images of a **straight** road (`straight_lines1/2.jpg`) I placed a
trapezoid (`src`, red in the figure) on the left and right line of our lane, from just above the car hood
to some distance ahead. It is mapped to a rectangle (`dst`) where our lane is between 35 % and 65 % of
the image width, so there is room left for the neighbouring lanes:

| | top-left | top-right | bottom-right | bottom-left |
|---|---|---|---|---|
| `src` (1280 x 720) | (567, 468) | (712, 468) | (1025, 676) | (254, 676) |
| `dst` | (448, 0) | (832, 0) | (832, 720) | (448, 720) |

**Check.** On a straight road both lines must become vertical and parallel. They lie on the red vertical
guides at 35 % and 65 %:

![alt text][image4]

The cameras in the recordings are not mounted the same way, so one trapezoid does not fit all of them.
I chose the points separately for each camera setup, always on a straight part of the road (the third row
above is the 960 x 540 camera).

**Region of interest (ROI).** The perspective transform only changes the geometry, it does not remove the
sky and trees from step 2. For that I use a ROI mask (blue in the figure): only the road in front of the
car is kept (`cv2.fillPoly` + `cv2.bitwise_and`).

From now on the image is first warped and then turned into a binary image (same functions as in step 2),
and the ROI is applied. In the top view all markings have about the same width, so one top-hat kernel
fits the whole image. Only the lane lines remain:

![alt text][image4b]

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
