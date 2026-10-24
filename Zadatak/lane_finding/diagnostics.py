
import cv2
import numpy as np

from .drawing import to_bgr
from .lane_lines import LINE_SLOTS, poly_x


GREEN, RED, YELLOW, CYAN, BLUE = (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0), (255, 0, 0)


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



def tracking_image(binary, lines, accepted, fits):
    """Like search_image, but for the video pipeline: rejected candidates are red,
    pixels found around the previous line are blue, the tracked lines are yellow."""
    img = to_bgr(binary) // 2
    ys = np.arange(binary.shape[0], dtype=float)
    for line, ok in zip(lines, accepted):
        for x1, y1, x2, y2, good in line.windows:
            cv2.rectangle(img, (x1, y1), (x2, y2), (GREEN if good else (0, 0, 128)) if ok else (0, 0, 90), 2)
        if len(line.xs):
            img[line.ys.astype(int), line.xs.astype(int)] = (CYAN if line.windows else BLUE) if ok else RED
    for fit in fits:
        if fit is not None:
            cv2.polylines(img, [np.stack([poly_x(fit, ys), ys], 1).astype(np.int32)], False, YELLOW, 4)
    _text(img, ["Search and tracked lines"])
    return img


def diagnostic_view(r, frame_index=None):
    """Six panels for one video frame (r is a pipeline.FrameResult):
    result | top view + ROI / binary | search / histogram | top view with the fits."""
    h, w = r.image.shape[:2]
    result = r.image.copy()
    if frame_index is not None:
        _text(result, [f"frame {frame_index}"], y0=330)
    warped = r.warped.copy()
    contours, _ = cv2.findContours(r.roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(warped, contours, -1, BLUE, 4)
    _text(warped, ["Top view + ROI"])
    binary = to_bgr(r.binary)
    _text(binary, ["Binary image"])
    overhead = r.warped.copy()
    for fit in r.fits:
        if fit is not None:
            ys = np.arange(h, dtype=float)
            cv2.polylines(overhead, [np.stack([poly_x(fit, ys), ys], 1).astype(np.int32)], False, (255, 0, 255), 8)
    _text(overhead, ["Top view with the lines"])
    panels = [result, warped, binary, tracking_image(r.binary, r.measurements, r.accepted, r.fits),
              histogram_image(r.hist, r.calculated_peaks, r.sorted_peaks, h), overhead]
    panels = [cv2.resize(p, (w // 2, h // 2)) for p in panels]
    return np.vstack([np.hstack(panels[i:i + 2]) for i in range(0, 6, 2)])

