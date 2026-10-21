
import cv2
import numpy as np

from .drawing import to_bgr
from .lane_lines import LINE_SLOTS, poly_x

GREEN, RED, YELLOW, CYAN = (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0)


def _text(img, lines, color=GREEN, y0=40):
    s = img.shape[1] / 640
    for i, t in enumerate(lines):
        cv2.putText(img, t, (int(12 * s), int((y0 + 30 * i) * s)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75 * s, color, max(1, int(2 * s)), cv2.LINE_AA)


def histogram_image(hist, calculated, sorted_peaks, h):
    """Histogram of the lower half of the binary image; red dots are the peaks,
    thin vertical lines the expected positions of Line0..Line3."""
    w = len(hist)
    img = np.zeros((h, w, 3), np.uint8)
    top = max(hist.max(), 1)
    pts = np.stack([np.arange(w), h - 1 - hist / top * 0.6 * h], axis=1).astype(np.int32)
    cv2.polylines(img, [pts], False, (255, 255, 255), 2)
    for slot in LINE_SLOTS:
        x = int(slot * w)
        cv2.line(img, (x, int(0.35 * h)), (x, h), (120, 120, 0), 1)
    for p in calculated:
        cv2.circle(img, (int(p), int(h - 1 - hist[p] / top * 0.6 * h)), 8, RED, -1)
    _text(img, ["Histogram with peaks"])
    _text(img, [f"Calculated peaks: {calculated}", f"Sorted peaks: {sorted_peaks}"], RED, 110)
    return img


def search_image(binary, lines):
    """Sliding windows (green = enough pixels, dark red = empty or too full),
    pixels of each line (cyan) and the fitted polynomials (yellow)."""
    img = to_bgr(binary) // 2
    ys = np.arange(binary.shape[0], dtype=float)
    for line in lines:
        for x1, y1, x2, y2, good in line.windows:
            cv2.rectangle(img, (x1, y1), (x2, y2), GREEN if good else (0, 0, 128), 2)
        if len(line.xs):
            img[line.ys.astype(int), line.xs.astype(int)] = CYAN
        if line.found:
            cv2.polylines(img, [np.stack([poly_x(line.fit, ys), ys], 1).astype(np.int32)], False, YELLOW, 4)
    _text(img, ["Sliding windows and fits"])
    return img
