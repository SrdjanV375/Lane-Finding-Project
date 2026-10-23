import glob
import os

import cv2

from lane_finding.calibration import load_calibration
from lane_finding.drawing import grid
from lane_finding.lane_image import LaneImage

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(ROOT, "..", "docs"))

PROFILE = {"challange00101": "highway_1280", "challange00111": "highway_1280",
           "challange00136": "highway_1280"}


def main():
    mtx, dist = load_calibration(os.path.join(ROOT, "camera_cal", "calib.npz"))
    os.makedirs(os.path.join(OUT, "test_results"), exist_ok=True)
    images, names = [], []
    for path in sorted(glob.glob(os.path.join(ROOT, "test_images", "*.jpg"))):
        name = os.path.splitext(os.path.basename(path))[0]
        img = cv2.imread(path)
        r = LaneImage((img.shape[1], img.shape[0]), mtx, dist, PROFILE.get(name)).process(img)
        cv2.imwrite(os.path.join(OUT, "test_results", name + ".jpg"), r.image)
        if r.radius_m is None:
            print(f"{name:20s} no lane found")
        else:
            radius = "> 3000" if r.radius_m > 3000 else f"{r.radius_m:.0f}"
            print(f"{name:20s} {r.status:9s} radius {radius:>7s} m   offset {r.offset_m:+.2f} m"
                  f"   lane width {r.width_m:.2f} m")
        images.append(r.image)
        names.append(name)
        if name == "test2":
            cv2.imwrite(os.path.join(OUT, "05_result_test2.jpg"), r.image)
    cv2.imwrite(os.path.join(OUT, "05_all_test_images.jpg"), grid(images, names, 3, 480))
    print("Saved results to docs/")


if __name__ == "__main__":
    main()
