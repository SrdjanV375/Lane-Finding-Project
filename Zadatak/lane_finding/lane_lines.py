"""Step 4 - finding lane-line pixels and fitting a polynomial.

Histogram peaks -> sliding windows -> polynomial x = A*y^2 + B*y + C fitted with RANSAC.

Expected line positions in the bird's-eye view (relative to the image width):
    Line0 ~ 0.05  left neighbouring lane
    Line1 ~ 0.35  left line of our lane
    Line2 ~ 0.65  right line of our lane
    Line3 ~ 0.95  right neighbouring lane
"""
from dataclasses import dataclass, field

import numpy as np

LINE_SLOTS = [0.05, 0.35, 0.65, 0.95]
SLOT_HALF_WIDTH = 0.13          # peak search range around each slot
N_WINDOWS = 9
MIN_PEAK = 10                   # min. smoothed histogram value of a peak
MIN_PEAK_DISTANCE = 0.15        # two lines are at least half a lane apart
MIN_CONFIDENCE = 0.3
INLIER_TOLERANCE = 0.015        # max. distance of an inlier pixel from the curve (relative to w)


@dataclass
class LineFit:
    fit: np.ndarray = None
    confidence: float = 0.0
    xs: np.ndarray = field(default_factory=lambda: np.empty(0))
    ys: np.ndarray = field(default_factory=lambda: np.empty(0))
    windows: list = field(default_factory=list)

    @property
    def found(self):
        return self.fit is not None and self.confidence >= MIN_CONFIDENCE


def poly_x(fit, y):
    return fit[0] * y ** 2 + fit[1] * y + fit[2]


def histogram(binary):
    """Number of white pixels in each column of the lower half of the image."""
    h = binary.shape[0]
    return np.sum(binary[h // 2:, :] > 0, axis=0)


def find_peaks(hist):
    w = len(hist)
    smooth = np.convolve(hist, np.ones(15) / 15, mode="same")
    sorted_peaks = []
    for slot in LINE_SLOTS:
        lo = max(0, int((slot - SLOT_HALF_WIDTH) * w))
        hi = min(w, int((slot + SLOT_HALF_WIDTH) * w))
        x = lo + int(np.argmax(smooth[lo:hi]))
        sorted_peaks.append(x if smooth[x] >= MIN_PEAK else None)
    # On a sharp curve two neighbouring slots can pick the same line: keep the closer one.
    for i in range(len(sorted_peaks) - 1):
        a, b = sorted_peaks[i], sorted_peaks[i + 1]
        if a is not None and b is not None and b - a < MIN_PEAK_DISTANCE * w:
            if abs(a - LINE_SLOTS[i] * w) <= abs(b - LINE_SLOTS[i + 1] * w):
                sorted_peaks[i + 1] = None
            else:
                sorted_peaks[i] = None
    calculated = sorted({p for p in sorted_peaks if p is not None})
    return calculated, sorted_peaks


def ransac_polyfit(ys, xs, h, w, iterations=60, tolerance=0.015, seed=0):
    """Parabola through 3 random pixels, repeated; the one with most inliers is refitted on them."""
    if len(ys) < 3 or np.ptp(ys) < 0.2 * h:
        return None
    rng = np.random.default_rng(seed)
    best_inliers = None
    for _ in range(iterations):
        idx = rng.choice(len(ys), 3, replace=False)
        y3 = ys[idx]
        if np.min(np.diff(np.sort(y3))) < 0.05 * h:    # points too close in y
            continue
        fit = np.polyfit(y3, xs[idx], 2)
        inliers = np.abs(poly_x(fit, ys) - xs) < tolerance * w
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    if best_inliers is None or best_inliers.sum() < 3:
        return np.polyfit(ys, xs, 2)
    return np.polyfit(ys[best_inliers], xs[best_inliers], 2)


def fit_line(result, ys, xs, h, w, tolerance=INLIER_TOLERANCE):
    """RANSAC fit; only pixels close to the final curve are kept and counted as support."""
    result.fit = ransac_polyfit(ys, xs, h, w, tolerance=tolerance)
    if result.fit is None:
        result.xs, result.ys, result.confidence = np.empty(0), np.empty(0), 0.0
        return result
    inliers = np.abs(poly_x(result.fit, ys) - xs) < tolerance * w
    result.ys, result.xs = ys[inliers], xs[inliers]
    result.confidence = coverage(result.ys, h)
    return result


def coverage(ys, h, min_pixels=20):
    if len(ys) == 0:
        return 0.0
    bands = np.minimum((ys * N_WINDOWS // h).astype(int), N_WINDOWS - 1)
    counts = np.bincount(bands, minlength=N_WINDOWS)
    return float(np.mean(counts >= min_pixels))


def sliding_window_polyfit(binary, x_base, margin_rel=0.05, minpix=40):
    """Sliding windows from the bottom up; each window is centred on the median x of the
    previous one. A window more than 40 % full is not a thin line (car, grass) and is skipped."""
    h, w = binary.shape
    nonzero_y, nonzero_x = binary.nonzero()
    margin = int(margin_rel * w)
    win_h = h // N_WINDOWS

    x_current = float(x_base)
    step = 0.0                # change of x from one window to the next
    last_good = None          # (window index, x) of the last good window
    misses = 0                # empty windows since the last good one
    keep = []
    windows = []
    for i in range(N_WINDOWS):
        y_low, y_high = h - (i + 1) * win_h, h - i * win_h
        # after a gap (dashed line) the next window is wider, at most twice
        half = margin * min(1 + 0.5 * misses, 2.0)
        x_low, x_high = int(x_current - half), int(x_current + half)
        idx = ((nonzero_y >= y_low) & (nonzero_y < y_high) &
               (nonzero_x >= x_low) & (nonzero_x < x_high)).nonzero()[0]
        good = minpix <= len(idx) <= 0.4 * (2 * half) * win_h
        windows.append((x_low, y_low, x_high, y_high, good))
        if good:
            keep.append(idx)
            x_new = float(np.median(nonzero_x[idx]))
            if last_good is not None:
                step = np.clip((x_new - last_good[1]) / (i - last_good[0]), -0.75 * margin, 0.75 * margin)
            last_good = (i, x_new)
            x_current = x_new + step
            misses = 0
        else:
            x_current += step     # keep the direction of the line through the gap
            misses += 1

    result = LineFit(windows=windows)
    if not keep:
        return result
    idx = np.concatenate(keep)
    return fit_line(result, nonzero_y[idx], nonzero_x[idx], h, w)


def find_lines(binary):
    """Find all four lines in one binary bird's-eye image."""
    hist = histogram(binary)
    calculated, sorted_peaks = find_peaks(hist)
    lines = [sliding_window_polyfit(binary, x) if x is not None else LineFit()
             for x in sorted_peaks]
    return lines, hist, calculated, sorted_peaks
