# Lane Finding Project

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
[image5]: ./docs/04_lane_fit.jpg "Fit Visual"
[image6]: ./docs/05_result_test2.jpg "Output"
[image6b]: ./docs/05_all_test_images.jpg "All test images"
[image7]: ./docs/06_video_project_video03.jpg "Video frames"
[image7b]: ./docs/06_video_challenge03.jpg "Challenge video frames"
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

**Code.** [lane_finding/calibration.py](Zadatak/lane_finding/calibration.py) (`calibrate_camera`, `undistort`),
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

**Skipped images.** In `calibration1.jpg`, `calibration4.jpg` and `calibration5.jpg` the complete
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

**Goal.** Find which white pixels belong to the left and right line of our lane and describe each line
with a curve.

**Code.** [lane_finding/lane_lines.py](Zadatak/lane_finding/lane_lines.py) (`find_lines`), run with
`python lane_fit_step.py`. The input is the binary top view with the ROI from step 3.

**Why this approach.** A common way to find lines is edge detection (Canny) followed by the Hough
transform. It finds **straight** lines, which works on a straight road, but a lane in a curve is not a
straight line, and the curvature in the next step cannot be measured from straight lines. So I chose a
method that follows the line as it bends: a **histogram** to find where the lines start, **sliding windows**
to collect their pixels, and a **second-order polynomial** to describe each line.

**1. Histogram - where do the lines start?** I count the white pixels in every column of the lower half of
the image; a lane line gives a high peak. I look for a peak near four expected positions: our two lines at
35 % and 65 % of the width (where step 3 puts them) and the neighbouring lines at 5 % and 95 %. A position
without a peak gets `None` (`test5`: `[None, 441, 857, 1237]`, no line far left).

**2. Sliding window - follow the line upwards.** From each peak, 9 windows go from the bottom to the top.
The white pixels in a window belong to the line, and the next window is moved to their centre. A **dashed
line** has gaps: after an empty window the next one keeps the direction of the line and is a bit wider. A
window that is almost full of white is not a thin line (a car, grass) and is ignored.

**3. Fit a curve.** The pixels are fitted with `x = A*y^2 + B*y + C` (x as a function of y, because the lines
are almost vertical). A normal least-squares fit is pulled away by a few wrong pixels, so I use **RANSAC**:
pick 3 random pixels many times, draw the parabola through them, count the pixels close to it, and refit
the best one using only those pixels.

The **support** of a line is the share of the 9 height bands where it really has pixels (solid line ~100 %,
dashed 50-80 %). Below 30 % the line is not accepted.

Example output (`test5`):

```
left : x = 9.20e-05*y^2 + -0.120*y + 479.3   support 78%
right: x = 1.50e-05*y^2 + -0.063*y + 889.1   support 78%
```

Histogram with peaks (left) and windows with line pixels (cyan) and fitted curves (yellow) on `test2`,
`test5` and `solidYellowCurve`; dark red windows are the gaps of the dashed line:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.


**Goal.** From the two lane-line polynomials, compute how sharply the road bends (radius of curvature)
and how far the car is from the centre of its lane, both in meters.

**Code.** [lane_finding/measurements.py](Zadatak/lane_finding/measurements.py) (`curvature_radius`,
`vehicle_offset`), used by [lane_finding/lane_image.py](Zadatak/lane_finding/lane_image.py); run with
`python images_step.py`.

**From pixels to meters.** The polynomials are in pixels of the top view, so I need to know how many
meters one pixel is in each direction:
* **across the road (x):** our lane is always 30 % of the image width in the top view (step 3), and a
  highway lane is 3.7 m wide, so `xm = 3.7 / (0.30 * width)` meters per pixel;
* **along the road (y):** I measured how long a piece of road the top view shows. Dashed lane lines repeat
  every 12.2 m (3 m line + 9 m gap). In the top view of a straight road I found how many pixels one
  dash + gap takes (`measure_view_length.py`, results in [docs/view_length.md](docs/view_length.md)):
  the view shows about **19-21 m** of road for all cameras, so I use 20 m, i.e. `ym = 20 / height`.

My first guess was 30 m; that would have made the radius up to 2.25 times too large, so measuring it was
worth it.

**Radius of curvature.** The pixel polynomial `x = A*y^2 + B*y + C` is converted to meters
(`A_m = A * xm / ym^2`, `B_m = B * xm / ym`) and the radius is computed at the bottom of the image, where
the car is:

```
R = (1 + (2*A_m*y + B_m)^2)^(3/2) / |2*A_m|
```

The lane radius is the average of the left and the right line. When the road is almost straight, `A` is
almost 0 and the radius becomes very large and unstable, so above 3000 m it is shown as "straight".

**Vehicle position.** The camera is in the middle of the car, so the car centre is the middle of the
camera image. I project that point into the top view (`Perspective.vehicle_x`) and compare it with the
middle between the two lines at the bottom of the image:

```
offset = (car_x - (left_x + right_x) / 2) * xm        (negative = car is left of the lane centre)
```

Example output (`python images_step.py`):

```
test2       measured  radius     572 m   offset -0.43 m   lane width 3.73 m
test5       measured  radius    1555 m   offset -0.06 m   lane width 3.96 m
```

The lane width is also printed as a check: on all test images it is 3.6-4.0 m, close to the 3.7 m used
for the x scale. These values are estimates - the scale depends on the assumed lane width and on a flat
road.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

**Goal.** Show the found lane on the original (undistorted) image, together with the radius and the
vehicle position.

**Code.** [lane_finding/lane_image.py](Zadatak/lane_finding/lane_image.py) (`LaneImage.process`,
`draw`), run with `python images_step.py`. It connects all steps for one image: undistort -> top view ->
binary image + ROI -> lane lines -> checks -> measurements -> drawing.

**How it is drawn.** The area between the two polynomials is filled in the **top view** and then warped
back into the camera view with the inverse matrix `Minv` from step 3, and blended with the image. The
lane is drawn only up to the height where the lines were really seen, not further.

* **green** - our lane, both lines found;
* **orange** - only one line was found; the other is drawn parallel to it, one lane width away;
* **red** - a neighbouring lane in the same direction (when its outer line was also found).

**Checks before drawing.** Not every line the search returns is accepted:
* the two lines of our lane must form a realistic lane (2.8-4.6 m wide and roughly parallel);
* a line with only a few dashes takes the shape of the better line, because lane lines are parallel -
  otherwise its curve bends where it has no pixels;
* in these recordings yellow lines are always on the left edge of our direction, and solid white lines on
  the right edge of the road. So a yellow line is never accepted as the right line of our lane, and no
  "neighbouring lane" is drawn behind a yellow or a solid line - that area is the shoulder or grass.
  This is an assumption about the markings in this data set, not a general rule.

Result on `test2.jpg`:

![alt text][image6]

**A mistake I found here.** The three `challange001xx` images are 960 x 540, so at first they got the
perspective points of the 960 x 540 camera. In the top view the lane on two of them became narrower towards the top
(by up to 16 %), which is wrong. Looking closer, they show the same bridge and the same cars as `test1`: they
are frames of the 1280 x 720 camera, only resized. With the points of that camera the lines are parallel
again (width change about 2 %), and the radius on `challange00136` changed from 2371 m to 551 m - close to
`test1` (510 m), which is the same road. So for these images the camera setup is now set explicitly
(`images_step.py`) instead of being guessed from the image size.

All 17 test images, each processed on its own:

![alt text][image6b]


Each result in full size is in [docs/test_results](docs/test_results).

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).


**Result videos** (click the name; GitHub shows smaller videos directly, larger ones with "View raw" or as
a download). The numbers are frames per state of the algorithm:

| Video | Frames | measured | held | inferred | missing | Warning frames |
|---|---|---|---|---|---|---|
| [project_video01](docs/videos/project_video01.mp4) | 221 | 221 | 0 | 0 | 0 | 0 |
| [project_video02](docs/videos/project_video02.mp4) | 681 | 681 | 0 | 0 | 0 | 0 |
| [project_video03](docs/videos/project_video03.mp4) | 1260 | 1259 | 1 | 0 | 0 | 0 |
| [challenge01](docs/videos/challenge01.mp4) | 251 | 251 | 0 | 0 | 0 | 0 |
| [challenge02](docs/videos/challenge02.mp4) | 484 | 444 | 15 | 20 | 5 | 0 |
| [challenge03](docs/videos/challenge03.mp4) | 1199 | 913 | 143 | 126 | 17 | 14 |

*measured* = both lines found in this frame, *held* = a line kept from previous frames, *inferred* = one
line drawn parallel to the other, *missing* = no lane shown. This is **not an accuracy**: a "measured"
frame can still be wrong (e.g. a guardrail taken as a line), and a "held" frame can look correct. The
quality was checked by looking at the result videos.

**Code.** [lane_finding/pipeline.py](Zadatak/lane_finding/pipeline.py) (`LaneFinder`),
[lane_finding/tracking.py](Zadatak/lane_finding/tracking.py), [process_video.py](Zadatak/process_video.py);
all six videos are processed with `python videos_step.py`. Every video also gets a CSV
file with one row per frame (state of the lane, radius, offset, lane width, support of each line).

**Why a single-image method is not enough.** Processing every frame on its own works, but the lane jumps
from frame to frame, and in a frame with a shadow, glare or a car over the dashes a line is simply lost.
In a video the lines move only a little between two frames (33-40 ms), so the previous frame helps a lot.

**What is added for video:**
* **Search around the previous line.** When a line was found in the previous frame, its pixels are
  searched only in a narrow band around it (`polyfit_using_prev_fit`). Only when that fails, the sliding
  window from step 4 starts again.
* **Kalman filter for each line.** Each line is described by its position at three heights of the top
  view. In every frame the filter first *predicts* the line (it stays where it was), then *corrects* it
  with the new measurement. A measurement with little support (a few dashes) moves the line less than a
  solid line, so the lane does not jump.
* **Holding a lost line.** If a line is not found, it is kept from the previous frames for at most 8
  frames (~0.3 s) and drawn in **orange**; after that it is dropped, so an old line is never shown for long.
  If only one line is available, the other is drawn parallel to it at the last measured lane width.
* **Checks between frames.** A line may not jump sideways more than a few percent of the image width
  between frames, and the lane width may not change suddenly - such a "line" is usually a guardrail or the
  edge of the road.
* **Lane departure warning.** When the edge of the car comes closer than 0.15 m to a lane line for 5
  frames in a row, the lane is drawn red with "Alert! Lane departure!".

**How it works on the videos.** On the three project videos and `challenge01` the lane is measured in
practically every frame, also on concrete, in shadows and with cars changing lanes. In `challenge02` the
lane is lost for about one second under the bridge (dark shadow), held for a moment, and then found again.
`challenge03` (mountain road) is much harder: sharp turns where one line leaves the camera view, strong
glare when driving out of the forest, and no dashed lines. For this video the camera points look less far
ahead, and two extra rules use the road markings (the left line of our lane is the yellow middle line).
It follows the road on most of the video, but not everywhere - see the discussion.

Sample frames from `project_video03` (concrete, shadows, cars in the next lane):

![alt text][image7]

Sample frames from `challenge03`:

![alt text][image7b]

Sample frames of all videos are in `docs/06_video_<name>.jpg`. A diagnostic video with six panels (result,
top view with ROI, binary image, search, histogram and fitted lines) can be made with
`python videos_step.py --diag`; these videos are not in the repository because of their size.

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

TODO: Add your text here!!!
