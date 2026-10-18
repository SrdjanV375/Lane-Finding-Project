import glob
import os

import cv2
import numpy as np

COLS = 9
ROWS = 6

CALIB_SIZE = (1280, 720)

def find_corners(gray):

    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)
    ret, corners = cv2.findChessboardCorners(gray, (COLS, ROWS), None)
    if not ret:
        return None
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


def calibrate_camera(image_paths):

    object_points = np.zeros((ROWS * COLS, 3), np.float32)
    object_points[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2)

    object_points_array = []
    image_points_array = []
    used = []

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            raise OSError(f"Cannot read {path}")
        img = cv2.resize(img, CALIB_SIZE)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners = find_corners(gray)
        if corners is None:
            # board partially outside the image (calibration1, 4 and 5)
            print(f"  {os.path.basename(path)}: 9x6 corners not found - skipped")
            continue
        object_points_array.append(object_points)
        image_points_array.append(corners)
        used.append(path)

    if len(used) < 3:
        raise ValueError("Not enough calibration images with a visible chessboard")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points_array, image_points_array, CALIB_SIZE, None, None)

    exercise, distances = 0.0, []
    for i in range(len(object_points_array)):
        projected, _ = cv2.projectPoints(object_points_array[i], rvecs[i], tvecs[i], mtx, dist)
        exercise += cv2.norm(image_points_array[i], projected, cv2.NORM_L2) / len(projected)
        distances.append(np.linalg.norm(image_points_array[i].reshape(-1, 2) - projected.reshape(-1, 2), axis=1))
    errors = {"rms": float(ret),
              "mean_corner": float(np.mean(np.concatenate(distances))),
              "exercise": exercise / len(object_points_array)}
    return mtx, dist, used, errors


def save_calibration(path, mtx, dist):
    np.savez(path, mtx=mtx, dist=dist)


def load_calibration(path):
    data = np.load(path)
    return data["mtx"], data["dist"]


def undistort(img, mtx, dist):
    if (img.shape[1], img.shape[0]) != CALIB_SIZE:
        return img
    return cv2.undistort(img, mtx, dist, None, mtx)


def calibration_images(folder):
    return sorted(glob.glob(os.path.join(folder, "calibration*.jpg")))
