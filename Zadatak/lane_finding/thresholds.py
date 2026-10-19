import cv2
import numpy as np

# Yellow in HSV (OpenCV: H 0-180, S and V 0-255)
YELLOW_LOW = np.array([15, 80, 100])
YELLOW_HIGH = np.array([35, 255, 255])

# Region of the road in front of the car (relative), used for the R histogram.
ROAD_Y = (0.65, 0.92)
ROAD_X = (0.25, 0.75)

R_PERCENTILE = 90       # "bright asphalt" level
R_ABOVE_ROAD = 25       # a marking is at least this much brighter than bright asphalt

R_MIN, R_MAX = 90, 245

TOPHAT_MIN = 20         # minimum local contrast of a marking (with the histogram threshold)
STRONG_TOPHAT = 40      # local contrast that is enough on its own (shadow, concrete)
WHITE_MAX_SAT = 60      # white markings are unsaturated (HSV S, 0-255)

# Yellow in Lab: b channel (128 = neutral grey, higher = more yellow).
YELLOW_B_MIN = 130
YELLOW_B_CONTRAST = 10


def yellow_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)


def yellow_lab_mask(img, kernel_width):
    """Yellow as a narrow stripe with higher Lab b than its surroundings."""
    b = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 2]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    contrast = cv2.morphologyEx(b, cv2.MORPH_TOPHAT, kernel)
    return cv2.bitwise_and(cv2.inRange(b, YELLOW_B_MIN, 255),
                           cv2.inRange(contrast, YELLOW_B_CONTRAST, 255))


def low_saturation_mask(img, max_sat=WHITE_MAX_SAT):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv[:, :, 1], 0, max_sat)


def red_threshold(img, road_y=ROAD_Y, road_x=ROAD_X):

    h, w = img.shape[:2]
    r = img[:, :, 2]   # OpenCV stores images as BGR -> index 2 is R
    road = r[int(road_y[0] * h):int(road_y[1] * h), int(road_x[0] * w):int(road_x[1] * w)]
    return float(np.clip(np.percentile(road, R_PERCENTILE) + R_ABOVE_ROAD, R_MIN, R_MAX))


def red_mask(img, **road):
    _, mask = cv2.threshold(img[:, :, 2], red_threshold(img, **road), 255, cv2.THRESH_BINARY)
    return mask


def tophat_mask(img, kernel_width, min_contrast=TOPHAT_MIN):
    """Narrow bright structures (lane markings) on the R channel."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    tophat = cv2.morphologyEx(img[:, :, 2], cv2.MORPH_TOPHAT, kernel)
    _, mask = cv2.threshold(tophat, min_contrast, 255, cv2.THRESH_BINARY)
    return mask


def sobel_x_mask(img, low=30, high=150):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    return cv2.inRange(sobel, low, high)


def binary_image(img, kernel_width=None, **road):
    """Final binary image: yellow mask OR white mask (see lane_masks).

    kernel_width - width of the top-hat kernel in pixels; it must be wider than
    a lane marking. In the bird's-eye view all markings have about the same
    width, so the same kernel works for the whole image.
    """
    yellow, white = lane_masks(img, kernel_width, **road)
    return cv2.bitwise_or(yellow, white)


def lane_masks(img, kernel_width=None, **road):
    """(yellow, white) masks; their union is the binary image."""
    blur = cv2.GaussianBlur(img, (5, 5), 0)    # noise removal (02_02_low_pass.py)
    if kernel_width is None:
        kernel_width = int(0.04 * img.shape[1]) | 1
    low_sat = low_saturation_mask(blur)
    white = cv2.bitwise_and(red_mask(blur, **road), tophat_mask(blur, kernel_width))
    white = cv2.bitwise_and(white, low_sat)
    strong = cv2.bitwise_and(tophat_mask(blur, kernel_width, STRONG_TOPHAT), low_sat)
    strong = cv2.bitwise_and(strong, cv2.inRange(blur[:, :, 2], R_MIN, 255))
    white = cv2.bitwise_or(white, strong)
    return (remove_wide_regions(yellow_lab_mask(blur, kernel_width), kernel_width),
            remove_wide_regions(white, kernel_width))


def remove_wide_regions(binary, kernel_width):
    """Remove white regions wider than kernel_width (dry grass, vegetation, cars).

    Opening with a horizontal kernel keeps only regions at least kernel_width
    wide; subtracting them leaves the narrow stripes (binary top-hat).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    wide = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return cv2.subtract(binary, wide)
