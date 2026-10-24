"""Pipeline (video) - process all six test videos.

    python videos_step.py
    python videos_step.py --diag

Writes docs/videos/<name>.mp4 and <name>.csv, docs/06_video_<name>.jpg (sample frames)
and prints a table with the state of the lane in all frames of every video.
"""
import argparse
import csv
import os

import cv2
import numpy as np

from lane_finding.drawing import grid
from process_video import process_video

ROOT = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.normpath(os.path.join(ROOT, "..", "docs"))

# camera setup of every video (perspective points, lane_finding/perspective.py)
VIDEOS = {
    "project_video01": "highway_960",
    "project_video02": "highway_960",
    "project_video03": "highway_1280",
    "challenge01": "highway_1280",
    "challenge02": "low_camera",
    "challenge03": "mountain",
}
STATES = ("measured", "held", "inferred", "missing")


def sample_frames(name, n=6):
    cap = cv2.VideoCapture(os.path.join(DOCS, "videos", name + ".mp4"))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tiles, titles = [], []
    for i in np.linspace(0, total - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if ok:
            tiles.append(frame)
            titles.append(f"frame {i}")
    cap.release()
    cv2.imwrite(os.path.join(DOCS, f"06_video_{name}.jpg"), grid(tiles, titles, 3, 480))


def summary_row(name):
    with open(os.path.join(DOCS, "videos", name + ".csv")) as f:
        rows = list(csv.DictReader(f))
    counts = [sum(r["status"] == s for r in rows) for s in STATES]
    alerts = sum(int(r["alert"]) for r in rows)
    return (f"| [{name}](docs/videos/{name}.mp4) | {len(rows)} | "
            + " | ".join(str(c) for c in counts) + f" | {alerts} |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diag", action="store_true")
    args = parser.parse_args()
    for name, profile in VIDEOS.items():
        process_video(os.path.join(ROOT, "test_videos", name + ".mp4"), profile, args.diag)
        sample_frames(name)
    print()
    print("| Video | Frames | " + " | ".join(STATES) + " | Warning frames |")
    print("|---|---|" + "---|" * len(STATES) + "---|")
    for name in VIDEOS:
        print(summary_row(name))


if __name__ == "__main__":
    main()
