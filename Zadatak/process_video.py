"""Run the pipeline on a video.

    python process_video.py test_videos/project_video01.mp4
    python process_video.py test_videos/challenge02.mp4 --profile low_camera --diag

Writes docs/videos/<name>.mp4, docs/videos/<name>.csv (one row per frame)
and with --diag also docs/videos/<name>_diag.mp4 (six-panel diagnostic view).
"""
import argparse
import csv
import os
import time

import cv2

from lane_finding.calibration import load_calibration
from lane_finding.diagnostics import diagnostic_view
from lane_finding.perspective import PROFILES_SRC
from lane_finding.pipeline import LaneFinder

ROOT = os.path.dirname(os.path.abspath(__file__))


def process_video(path, profile=None, diag=False, out_dir=None):
    out_dir = out_dir or os.path.normpath(os.path.join(ROOT, "..", "docs", "videos"))
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(path))[0]
    mtx, dist = load_calibration(os.path.join(ROOT, "camera_cal", "calib.npz"))

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise OSError(f"Cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(os.path.join(out_dir, name + ".mp4"), fourcc, fps, (w, h))
    out_diag = cv2.VideoWriter(os.path.join(out_dir, name + "_diag.mp4"), fourcc, fps,
                               (w // 2 * 2, h // 2 * 3)) if diag else None

    if not out.isOpened() or (out_diag is not None and not out_diag.isOpened()):
        cap.release()
        raise OSError("Cannot open the output video (codec or output folder problem)")

    finder = LaneFinder((w, h), mtx, dist, profile, video=True)
    rows = []
    start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = finder.process(frame)
        out.write(r.image)
        if out_diag is not None:
            out_diag.write(diagnostic_view(r, len(rows)))
        rows.append({"frame": len(rows), "status": r.status,
                     "radius_m": "" if r.radius_m is None else round(r.radius_m, 1),
                     "offset_m": "" if r.offset_m is None else round(r.offset_m, 3),
                     "lane_width_m": "" if r.width_m is None else round(r.width_m, 3),
                     "alert": int(r.alert),
                     **{f"conf{i}": round(c, 2) for i, c in enumerate(r.confidences)},
                     **{f"state{i}": s for i, s in enumerate(r.states)}})
    cap.release()
    out.release()
    if out_diag is not None:
        out_diag.release()
    if not rows:
        raise RuntimeError(f"No frames could be read from {path}")
    expected = int(cv2.VideoCapture(path).get(cv2.CAP_PROP_FRAME_COUNT))
    if expected > 0 and len(rows) != expected:
        raise RuntimeError(f"{name}: decoded {len(rows)} of {expected} frames")
    written = cv2.VideoCapture(os.path.join(out_dir, name + ".mp4"))
    if int(written.get(cv2.CAP_PROP_FRAME_COUNT)) != len(rows):
        raise RuntimeError(f"{name}: the written video does not contain all {len(rows)} frames")
    written.release()

    with open(os.path.join(out_dir, name + ".csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    counts = {s: sum(r["status"] == s for r in rows) for s in ("measured", "held", "inferred", "missing")}
    print(f"{name}: {len(rows)} frames, profile={finder.profile}, {time.time() - start:.0f} s, {counts}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="+")
    parser.add_argument("--profile", choices=sorted(PROFILES_SRC))
    parser.add_argument("--diag", action="store_true", help="also write the diagnostic video")
    args = parser.parse_args()
    for v in args.videos:
        process_video(v, args.profile, args.diag)
