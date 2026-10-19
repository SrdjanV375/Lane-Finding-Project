"""Step 2 - thresholded binary image on example images.

    python binary_step.py

Loads the calibration from step 1, undistorts each example image and builds the
binary image (lane_finding/thresholds.py). Writes:

    output_images/02_binary.jpg     the masks side by side for several examples
    output_images/02_binary_<name>.jpg   original and final binary image for each example
"""
import os

import cv2
import numpy as np

from lane_finding import thresholds as T
from lane_finding.calibration import load_calibration, undistort
from lane_finding.drawing import grid, side_by_side

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "output_images")

EXAMPLES = [("test5", "test_images/test5.jpg", None),
            ("straight_lines1", "test_images/straight_lines1.jpg", None),
            ("challenge03_frame60", "test_videos/challenge03.mp4", 60)]


def read(path, frame):
    full = os.path.join(ROOT, path)
    if frame is None:
        img = cv2.imread(full)
    else:
        cap = cv2.VideoCapture(full)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = cap.read()
        cap.release()
        img = img if ok else None
    if img is None:
        raise OSError(f"Cannot read {path}")
    return img


def main():
    mtx, dist = load_calibration(os.path.join(ROOT, "camera_cal", "calib.npz"))
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for name, path, frame in EXAMPLES:
        img = undistort(read(path, frame), mtx, dist)
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        yellow, white = T.lane_masks(img)
        final = cv2.bitwise_or(yellow, white)
        threshold = T.red_threshold(blur)
        print(f"{name:20s} R threshold from the road histogram: {threshold:.0f}")

        rows.append(grid([img, T.yellow_mask(blur), yellow, white, final, T.sobel_x_mask(blur)],
                         [name, "yellow HSV (first try)", "yellow Lab b", f"white (R > {threshold:.0f})",
                          "final binary", "Sobel x (not used)"], 6, 320))
        cv2.imwrite(os.path.join(OUT, f"02_binary_{name}.jpg"),
                    side_by_side([img, final], ["Undistorted", "Binary"]))

    cv2.imwrite(os.path.join(OUT, "02_binary.jpg"), np.vstack(rows))
    print("Saved binary images to output_images/")


if __name__ == "__main__":
    main()
