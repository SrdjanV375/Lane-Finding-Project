import os

import cv2
import numpy as np

from lane_finding.calibration import (
    calibrate_camera,
    calibration_images,
    save_calibration,
    undistort,
)
from lane_finding.drawing import side_by_side

ROOT = os.path.dirname(os.path.abspath(__file__))
CALIB_FILE = os.path.join(ROOT, "camera_cal", "calib.npz")
# images for the README are stored in /docs at the root of the repository
DOCS = os.path.normpath(os.path.join(ROOT, "..", "docs"))


def main():
    print("Camera calibration:")
    mtx, dist, used, errors = calibrate_camera(calibration_images(os.path.join(ROOT, "camera_cal")))
    save_calibration(CALIB_FILE, mtx, dist)

    np.set_printoptions(precision=4, suppress=True)
    print(f"Images used: {len(used)}")
    print(f"Reprojection error: RMS {errors['rms']:.3f} px, mean per corner {errors['mean_corner']:.3f} px "
          f"(09_01 formula L2/N: {errors['exercise']:.3f})")
    print("Camera matrix:\n", mtx)
    print("Distortion coefficients (k1, k2, p1, p2, k3):", dist.ravel())

    out = DOCS
    os.makedirs(out, exist_ok=True)
    board = cv2.imread(os.path.join(ROOT, "camera_cal", "calibration1.jpg"))
    if board is None:
        raise OSError("Cannot read camera_cal/calibration1.jpg")
    board = cv2.resize(board, (1280, 720))
    cv2.imwrite(os.path.join(out, "01_undistort_chessboard.jpg"),
                side_by_side([board, undistort(board, mtx, dist)], ["Original", "Undistorted"]))
    road = cv2.imread(os.path.join(ROOT, "test_images", "test1.jpg"))
    if road is None:
        raise OSError("Cannot read test_images/test1.jpg")
    cv2.imwrite(os.path.join(out, "01_undistort_road.jpg"),
                side_by_side([road, undistort(road, mtx, dist)], ["Original", "Undistorted"]))
    print("Saved", CALIB_FILE, "and example images to docs/")


if __name__ == "__main__":
    main()
