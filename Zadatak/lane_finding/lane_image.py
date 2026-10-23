from dataclasses import dataclass, field

import cv2
import numpy as np

from . import lane_lines as ll
from .calibration import undistort
from .measurements import curvature_radius, lane_width, vehicle_offset
from .perspective import LANE_WIDTH_REL, Perspective
from .thresholds import lane_masks

ROAD_REGION = dict(road_y=(0.3, 1.0), road_x=(0.38, 0.62))   # road in front of the car in the top view
MIN_LANE_M, MAX_LANE_M, MAX_WIDTH_CHANGE_M = 2.8, 4.6, 0.9    # a realistic lane
WEAK_LINE, STRONG_LINE = 0.6, 0.8      # a weak line takes the shape of a strong one
SOLID_MIN = 0.95                       # support of a solid (not dashed) line


@dataclass
class LaneResult:
    image: np.ndarray
    status: str = "missing"            # measured / inferred / missing
    radius_m: float = None
    offset_m: float = None
    width_m: float = None
    lines: list = None                 # the four line candidates (LineFit)
    accepted: list = None              # which candidates passed the checks
    left: np.ndarray = None
    right: np.ndarray = None
    binary: np.ndarray = None


class LaneImage:
    def __init__(self, img_size, mtx, dist, profile=None):
        self.w, self.h = img_size
        self.mtx, self.dist = mtx, dist
        self.p = Perspective(img_size, profile)
        self.profile = self.p.profile

    # ------------------------------------------------------------------ line search

    def find_candidates(self, binary):
        """Step 4 for all four slots, plus a second start point from the full-height histogram."""
        lines, hist, calculated, peaks = ll.find_lines(binary)
        full_peaks = ll.find_peaks(np.sum(binary > 0, axis=0))[1]
        for i, (peak, alt) in enumerate(zip(peaks, full_peaks)):
            # a dashed line can have its gap exactly in the lower half used by the histogram
            if alt is not None and (peak is None or abs(alt - peak) > 0.05 * self.w):
                retry = ll.sliding_window_polyfit(binary, alt)
                if retry.found and retry.confidence > lines[i].confidence:
                    lines[i] = retry
        return lines

    def plausible_line(self, i, line):
        """The line must stay inside the image and near its expected position."""
        if not line.found:
            return False
        xs = ll.poly_x(line.fit, np.linspace(0, self.h - 1, 9))
        if np.any(xs < -0.3 * self.w) or np.any(xs > 1.3 * self.w):
            return False
        return abs(xs[-1] / self.w - ll.LINE_SLOTS[i]) <= 0.18

    def plausible_pair(self, left, right):
        """The two lines must form a realistic, roughly parallel lane."""
        ys = np.linspace(0, self.h - 1, 9)
        widths = np.array([lane_width(left, right, self.w, self.h, y) for y in ys])
        return bool(MIN_LANE_M <= widths[-1] <= MAX_LANE_M and
                    np.all((widths > MIN_LANE_M - 0.5) & (widths < MAX_LANE_M + 0.5)) and
                    np.ptp(widths) <= MAX_WIDTH_CHANGE_M)

    def is_yellow(self, fit, binary, yellow):
        ys, xs = binary.nonzero()
        near = np.abs(xs - ll.poly_x(fit, ys)) < 0.03 * self.w
        return near.sum() >= 50 and bool(np.mean(yellow[ys[near], xs[near]] > 0) > 0.5)

    def reshape(self, lines, valid, weak, strong):
        """Lane lines are parallel: the weak line keeps its position, takes the strong line's shape."""
        w_line = lines[weak]
        if not len(w_line.xs):
            return
        a, b, _ = lines[strong].fit
        c = float(np.median(w_line.xs - a * w_line.ys ** 2 - b * w_line.ys))
        w_line.fit = np.array([a, b, c])
        keep = np.abs(ll.poly_x(w_line.fit, w_line.ys) - w_line.xs) < ll.INLIER_TOLERANCE * self.w
        w_line.xs, w_line.ys = w_line.xs[keep], w_line.ys[keep]
        w_line.confidence = ll.coverage(w_line.ys, self.h)
        if not w_line.found:
            valid[weak] = False

    def check(self, lines, binary, yellow):
        """Which candidates are accepted (Line0..Line3)."""
        valid = [self.plausible_line(i, line) for i, line in enumerate(lines)]
        # our lane must be realistic, otherwise the weaker line is dropped
        if valid[1] and valid[2] and not self.plausible_pair(lines[1].fit, lines[2].fit):
            valid[1 if lines[1].confidence < lines[2].confidence else 2] = False
        # yellow is always on the left of our direction: never a right line
        for i in (2, 3):
            if valid[i] and self.is_yellow(lines[i].fit, binary, yellow):
                valid[i] = False
        # a line with a few dashes bends where it has no pixels: give it the other line's shape
        if valid[1] and valid[2]:
            weak, strong = (1, 2) if lines[1].confidence < lines[2].confidence else (2, 1)
            if lines[weak].confidence < WEAK_LINE and lines[strong].confidence >= STRONG_LINE:
                self.reshape(lines, valid, weak, strong)
        # behind a yellow or a solid line of our lane there is no lane in our direction
        for i, ego in ((0, 1), (3, 2)):
            if valid[i] and (not valid[ego] or lines[ego].confidence >= SOLID_MIN or
                             (i == 0 and self.is_yellow(lines[ego].fit, binary, yellow))):
                valid[i] = False
        # neighbouring lines are parallel to our lane as well
        ego = [i for i in (1, 2) if valid[i]]
        if ego:
            strong = max(ego, key=lambda i: lines[i].confidence)
            for i in (0, 3):
                if valid[i] and lines[i].confidence < lines[strong].confidence:
                    self.reshape(lines, valid, i, strong)
        return valid

    # ------------------------------------------------------------------ one image

    def process(self, img):
        undist = undistort(img, self.mtx, self.dist)
        yellow, white = lane_masks(self.p.warp(undist), **ROAD_REGION)
        binary = cv2.bitwise_and(cv2.bitwise_or(yellow, white), self.p.roi)

        lines = self.find_candidates(binary)
        valid = self.check(lines, binary, yellow)
        r = LaneResult(image=undist.copy(), lines=lines, accepted=valid, binary=binary)

        left = lines[1].fit if valid[1] else None
        right = lines[2].fit if valid[2] else None
        r.status = "measured" if left is not None and right is not None else "inferred"
        lane_px = LANE_WIDTH_REL * self.w          # only one line found: the other one is parallel
        if left is None and right is not None:
            left = right - np.array([0, 0, lane_px])
        elif right is None and left is not None:
            right = left + np.array([0, 0, lane_px])
        if left is None:
            r.status = "missing"
            return r

        r.left, r.right = left, right
        r.radius_m = float(np.mean([curvature_radius(f, self.w, self.h, self.profile) for f in (left, right)]))
        r.offset_m = vehicle_offset(left, right, self.p.vehicle_x(), self.w, self.h)
        r.width_m = float(lane_width(left, right, self.w, self.h))
        neighbours = [lines[i].fit for i in (0, 3) if valid[i] and r.status == "measured"]
        seen = [lines[i].ys.min() for i in (1, 2) if valid[i] and len(lines[i].ys)]
        r.image = self.draw(undist, left, right, neighbours, r, int(max(seen)) if seen else 0)
        return r

    # ------------------------------------------------------------------ drawing (step 6)

    def draw(self, undist, left, right, neighbours, r, y_top):
        ys = np.arange(y_top, self.h, dtype=float)

        def pts(fit):
            return np.stack([ll.poly_x(fit, ys), ys], axis=1)

        layer = np.zeros_like(undist)
        color = (0, 255, 0) if r.status == "measured" else (0, 165, 255)
        cv2.fillPoly(layer, [np.vstack([pts(left), pts(right)[::-1]]).astype(np.int32)], color)
        for nfit in neighbours:
            ref = left if ll.poly_x(nfit, self.h - 1) < ll.poly_x(left, self.h - 1) else right
            cv2.fillPoly(layer, [np.vstack([pts(nfit), pts(ref)[::-1]]).astype(np.int32)], (0, 0, 255))
        for fit in [left, right, *neighbours]:
            cv2.polylines(layer, [pts(fit).astype(np.int32)], False, (0, 255, 255), int(0.012 * self.w))
        out = cv2.addWeighted(undist, 1.0, self.p.unwarp(layer), 0.4, 0)

        s = self.w / 1280
        radius = "> 3000 m (straight)" if r.radius_m > 3000 else f"{r.radius_m:.0f} m"
        side = "right" if r.offset_m > 0 else "left"
        text = [f"Radius of curvature: {radius}", f"Vehicle is {abs(r.offset_m):.2f} m {side} of center"]
        shade = out.copy()
        cv2.rectangle(shade, (0, 0), (self.w, int(105 * s)), (0, 0, 0), -1)
        cv2.addWeighted(shade, 0.45, out, 0.55, 0, out)
        for i, t in enumerate(text):
            cv2.putText(out, t, (int(20 * s), int((45 + 40 * i) * s)), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0 * s, (255, 255, 255), max(1, int(2 * s)), cv2.LINE_AA)
        return out
