import os

import cv2
import numpy as np

from lane_finding.calibration import load_calibration, undistort
from lane_finding.perspective import Perspective
from lane_finding.lane_image import ROAD_REGION
from lane_finding.thresholds import binary_image

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(ROOT, "..", "docs"))
DASH_PERIOD_M = 12.19

# (source, frame index for videos or None for images, camera profile)
INPUTS = [("test_images/straight_lines1.jpg", None, None),
          ("test_images/straight_lines2.jpg", None, None),
          ("test_images/test3.jpg", None, None),
          ("test_images/solidWhiteRight.jpg", None, None),
          ("test_images/solidYellowLeft.jpg", None, None)] + \
         [("test_videos/challenge02.mp4", f, "low_camera") for f in range(315, 436, 15)]


def read(path, frame):
    if frame is None:
        return cv2.imread(os.path.join(ROOT, path))
    cap = cv2.VideoCapture(os.path.join(ROOT, path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()
    return img if ok else None


def dash_period(binary, x_rel):
    """Period (px) of the on/off pattern of the line strip around x_rel, or None."""
    h, w = binary.shape
    strip = binary[:, int((x_rel - 0.04) * w):int((x_rel + 0.04) * w)].max(axis=1) > 0
    painted = strip.mean()
    if not 0.15 < painted < 0.8:          # solid line or no line
        return None, painted
    s = strip.astype(float) - painted
    ac = np.correlate(s, s, "full")[h - 1:]
    # a plausible view is 10-40 m long -> the 12.19 m period is 0.3h - 1.2h; shorter lags
    # are the spacing of broken paint inside one dash, not the dash period
    lo, hi = int(0.3 * h), int(0.8 * h)
    lag = lo + int(np.argmax(ac[lo:hi]))
    if ac[lag] < 0.2 * ac[0]:              # no clear periodicity
        return None, painted
    return lag, painted


def main():
    mtx, dist = load_calibration(os.path.join(ROOT, "camera_cal", "calib.npz"))
    rows, values = [], {}
    for path, frame, profile in INPUTS:
        img = read(path, frame)
        if img is None:
            continue
        img = undistort(img, mtx, dist)
        h, w = img.shape[:2]
        p = Perspective((w, h), profile)
        binary = binary_image(p.warp(img), **ROAD_REGION)
        for side, x in (("left", 0.35), ("right", 0.65)):
            period, painted = dash_period(binary, x)
            if period is None:
                continue
            length = DASH_PERIOD_M * h / period
            name = os.path.basename(path) + ("" if frame is None else f" #{frame}")
            rows.append(f"| {name} | {p.profile} | {side} | {painted:.2f} | {period} | {length:.1f} |")
            values.setdefault(p.profile, []).append(length)

    lines = ["| Input | Profile | Dashed line | Painted share | Period (px) | View length (m) |",
             "|---|---|---|---|---|---|", *rows, "",
             "| Profile | n | Median (m) | Min - max (m) |", "|---|---|---|---|"]
    for profile, v in values.items():
        lines.append(f"| {profile} | {len(v)} | {np.median(v):.1f} | {min(v):.1f} - {max(v):.1f} |")
    text = "\n".join(lines)
    with open(os.path.join(OUT, "view_length.md"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
