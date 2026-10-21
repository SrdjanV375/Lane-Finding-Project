import os

import cv2
import numpy as np

from lane_finding import lane_lines as ll
from lane_finding import thresholds as T
from lane_finding.calibration import load_calibration, undistort
from lane_finding.diagnostics import histogram_image, search_image
from lane_finding.drawing import grid
from lane_finding.perspective import Perspective

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(ROOT, "..", "docs"))
ROAD_REGION = dict(road_y=(0.3, 1.0), road_x=(0.38, 0.62))

EXAMPLES = ["test2", "test5", "solidYellowCurve"]


def main():
    mtx, dist = load_calibration(os.path.join(ROOT, "camera_cal", "calib.npz"))
    os.makedirs(OUT, exist_ok=True)
    tiles, titles = [], []
    for name in EXAMPLES:
        img = cv2.imread(os.path.join(ROOT, "test_images", name + ".jpg"))
        if img is None:
            raise OSError(f"Cannot read test_images/{name}.jpg")
        img = undistort(img, mtx, dist)
        p = Perspective((img.shape[1], img.shape[0]))
        binary = cv2.bitwise_and(T.binary_image(p.warp(img), **ROAD_REGION), p.roi)

        lines, hist, calculated, sorted_peaks = ll.find_lines(binary)
        ego = lines[1:3]
        print(f"{name}: sorted peaks {sorted_peaks}")
        for label, line in zip(("left ", "right"), ego):
            if line.found:
                a, b, c = line.fit
                print(f"   {label}: x = {a:.2e}*y^2 + {b:.3f}*y + {c:.1f}   support {line.confidence:.0%}")
            else:
                print(f"   {label}: not found")

        tiles += [histogram_image(hist, calculated, sorted_peaks, binary.shape[0]),
                  search_image(binary, ego)]
        titles += [f"{name}: histogram", f"{name}: windows + fit"]
    cv2.imwrite(os.path.join(OUT, "04_lane_fit.jpg"), grid(tiles, titles, 2, 640))
    print("Saved docs/04_lane_fit.jpg")


if __name__ == "__main__":
    main()
