"""Step 3 - perspective transform (bird's-eye view) and region of interest.

    python perspective_step.py

Writes:
    docs/03_perspective.jpg     source trapezoid + ROI on straight roads, and the warped images
    docs/03_binary_warped.jpg   binary image in the camera view vs. in the bird's-eye view with ROI
"""
import os

import cv2
import numpy as np

from lane_finding import thresholds as T
from lane_finding.calibration import load_calibration, undistort
from lane_finding.drawing import grid
from lane_finding.perspective import Perspective

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(ROOT, "..", "docs"))

ROAD_REGION = dict(road_y=(0.3, 1.0), road_x=(0.38, 0.62))


def read(path, frame=None):
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


def perspective_figure(mtx, dist):
    tiles, titles = [], []
    for name in ("straight_lines1", "straight_lines2", "solidWhiteRight"):
        img = undistort(read(f"test_images/{name}.jpg"), mtx, dist)
        p = Perspective((img.shape[1], img.shape[0]))
        camera = img.copy()
        cv2.polylines(camera, [p.roi_polygon], True, (255, 0, 0), 3)          # ROI - blue
        cv2.polylines(camera, [p.src.astype(np.int32)], True, (0, 0, 255), 3)  # src trapezoid - red
        bev = p.warp(img)
        for x in (0.35, 0.65):                                                  # expected line positions
            cv2.line(bev, (int(x * p.w), 0), (int(x * p.w), p.h), (0, 0, 255), 2)
        tiles += [camera, bev]
        titles += [f"{name}: src (red), ROI (blue)", f"warped ({p.profile})"]
        print(f"{name:16s} profile {p.profile}, src points: {p.src.astype(int).tolist()}")
    cv2.imwrite(os.path.join(OUT, "03_perspective.jpg"), grid(tiles, titles, 2, 640))


def binary_figure(mtx, dist):

    rows = []
    for name, path, frame, profile in (("test5", "test_images/test5.jpg", None, None),
                                       ("challenge03_frame60", "test_videos/challenge03.mp4", 60, "mountain")):
        img = undistort(read(path, frame), mtx, dist)
        p = Perspective((img.shape[1], img.shape[0]), profile)
        camera_binary = T.binary_image(img)
        bev = p.warp(img)
        bev_binary = cv2.bitwise_and(T.binary_image(bev, **ROAD_REGION), p.roi)
        roi_view = bev.copy()
        contours, _ = cv2.findContours(p.roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(roi_view, contours, -1, (255, 0, 0), 4)
        rows.append(grid([camera_binary, roi_view, bev_binary],
                         [f"{name}: binary (step 2)", "bird's-eye view + ROI", "binary in bird's-eye view"],
                         3, 426))
    cv2.imwrite(os.path.join(OUT, "03_binary_warped.jpg"), np.vstack(rows))


def main():
    mtx, dist = load_calibration(os.path.join(ROOT, "camera_cal", "calib.npz"))
    os.makedirs(OUT, exist_ok=True)
    perspective_figure(mtx, dist)
    binary_figure(mtx, dist)
    print("Saved perspective images to docs/")


if __name__ == "__main__":
    main()
