from dataclasses import dataclass, field

import cv2
import numpy as np

from . import lane_lines as ll
from .calibration import undistort
from .measurements import curvature_radius, lane_width, meters_per_pixel, vehicle_offset
from .perspective import LANE_WIDTH_REL, Perspective
from .thresholds import lane_masks
from .tracking import LineKalman

# realistic lane: width at the car, max width change, from which height it is checked
# (on the mountain road only the near half, because the road is not flat)
PAIR_CHECK = {
    "default": dict(min_m=2.8, max_m=4.6, max_change_m=0.9, from_y=0.0),
    "mountain": dict(min_m=2.6, max_m=5.4, max_change_m=1.5, from_y=0.5),
}
# max sideways jump of a line between frames (relative to w), bigger after missed frames
MAX_JUMP_REL = 0.06
JUMP_GROWTH = 0.25

# the mountain road has only two lanes: no neighbouring lane, the left line is yellow
SHOW_NEIGHBOURS = {"mountain": False}
LEFT_LINE_MUST_BE_YELLOW = {"mountain": True}

# a solid line has pixels in all 9 bands, a dashed one has gaps
SOLID_MIN = 0.95
WEAK_LINE, STRONG_LINE = 0.6, 0.8

# the lane width is learned slowly; a sudden change of more than 15 % is suspicious
WIDTH_LEARN = 0.03
MAX_WIDTH_DEVIATION = 0.15

# lane departure warning: car edge closer than 0.15 m to a line for 5 frames
HALF_CAR_M = 0.9
DEPARTURE_MARGIN_M = 0.15
DEPARTURE_FRAMES = 5

# road in front of the car in the top view (for the histogram threshold)
ROAD_REGION = dict(road_y=(0.3, 1.0), road_x=(0.38, 0.62))


@dataclass
class FrameResult:
    image: np.ndarray
    status: str = "missing"
    radius_m: float = None
    offset_m: float = None
    width_m: float = None
    alert: bool = False
    confidences: list = field(default_factory=lambda: [0.0] * 4)
    states: list = field(default_factory=lambda: ["missing"] * 4)
    undistorted: np.ndarray = None
    warped: np.ndarray = None
    binary: np.ndarray = None
    hist: np.ndarray = None
    calculated_peaks: list = None
    sorted_peaks: list = None
    measurements: list = None
    fits: list = None
    mode: str = ""
    accepted: list = None
    roi: np.ndarray = None
    y_top: int = 0


class LaneFinder:
    def __init__(self, img_size, mtx, dist, profile=None, video=True):
        self.w, self.h = img_size
        self.mtx, self.dist = mtx, dist
        self.perspective = Perspective(img_size, profile)
        self.profile = self.perspective.profile
        self.video = video
        self.tracks = [LineKalman(self.h) for _ in ll.LINE_SLOTS]
        self.alert_count = 0
        # last measured lane width (px), used to draw a missing line
        self.lane_px = LANE_WIDTH_REL * self.w


    def _measure(self, binary, sorted_peaks):
        """Search each line around its tracked position, else with the sliding window."""
        results, modes = [], set()
        full_peaks = None
        for i, (track, peak) in enumerate(zip(self.tracks, sorted_peaks)):
            line = ll.LineFit()
            if self.video and track.fit is not None:
                line = ll.polyfit_using_prev_fit(binary, track.fit)
                modes.add("polyfit_using_prev_fit")
            if not line.found and peak is not None:
                line = ll.sliding_window_polyfit(binary, peak)
                modes.add("sliding_window_polyfit")
            # second start point: a dashed line can have its gap exactly in the lower half
            if not line.found or line.windows:
                if full_peaks is None:
                    full_peaks = ll.find_peaks(np.sum(binary > 0, axis=0))[1]
                alt = full_peaks[i]
                if alt is not None and (peak is None or abs(alt - peak) > 0.05 * self.w):
                    retry = ll.sliding_window_polyfit(binary, alt)
                    if retry.found and retry.confidence > line.confidence:
                        line = retry
                        modes.add("sliding_window_polyfit")
            results.append(line)
        if "sliding_window_polyfit" in modes:
            mode = "sliding_window_polyfit"
        else:
            mode = "polyfit_using_prev_fit" if modes else "no line candidates"
        return results, mode

    def _plausible_line(self, i, line):
        """A line must be inside the image, near its slot and must not jump."""
        if not line.found:
            return False
        xs = ll.poly_x(line.fit, np.linspace(0, self.h - 1, 9))
        if not np.all(np.isfinite(xs)) or np.any(xs < -0.3 * self.w) or np.any(xs > 1.3 * self.w):
            return False
        if abs(xs[-1] / self.w - ll.LINE_SLOTS[i]) > 0.18:
            return False
        track = self.tracks[i]
        # compare only where both lines were seen; the tolerance grows while the line is held
        if track.x is None:
            return True
        tolerance = MAX_JUMP_REL * self.w * (1 + JUMP_GROWTH * track.lost)
        y_from = max(float(line.ys.min()) if len(line.ys) else 0.0, track.y_top)
        return track.distance(line.fit, y_from) <= tolerance

    def plausible_pair(self, left, right):
        """The two lines must form a realistic lane."""
        c = PAIR_CHECK.get(self.profile, PAIR_CHECK["default"])
        ys = np.linspace(c["from_y"] * (self.h - 1), self.h - 1, 9)
        widths = np.array([lane_width(left, right, self.w, self.h, y) for y in ys])
        return bool(c["min_m"] <= widths[-1] <= c["max_m"] and
                    np.all((widths > c["min_m"] - 0.5) & (widths < c["max_m"] + 0.5)) and
                    np.ptp(widths) <= c["max_change_m"])


    def process(self, frame):
        if not self.video:
            for t in self.tracks:
                t.reset()

        undist = undistort(frame, self.mtx, self.dist)
        warped = self.perspective.warp(undist)
        yellow, white = lane_masks(warped, **ROAD_REGION)
        binary = cv2.bitwise_and(cv2.bitwise_or(yellow, white), self.perspective.roi)

        hist = ll.histogram(binary)
        calculated, sorted_peaks = ll.find_peaks(hist)
        measured, mode = self._measure(binary, sorted_peaks)
        valid = [self._plausible_line(i, m) for i, m in enumerate(measured)]

        if valid[1] and valid[2] and not self.plausible_pair(measured[1].fit, measured[2].fit):
            weaker = 1 if measured[1].confidence < measured[2].confidence else 2
            valid[weaker] = False
        self.check_candidates(measured, valid, binary, yellow)

        for track, line, ok in zip(self.tracks, measured, valid):
            track.predict()
            if ok:
                track.update(line.fit, line.confidence, float(line.ys.min()))
            else:
                track.miss()

        left, right = self.tracks[1].fit, self.tracks[2].fit
        states = [t.state for t in self.tracks]
        # a held line drifted away from the measured one
        if left is not None and right is not None and not self.plausible_pair(left, right):
            weaker = 1 if self.tracks[1].confidence < self.tracks[2].confidence else 2
            self.tracks[weaker].reset()
            states[weaker] = "missing"
            left, right = self.tracks[1].fit, self.tracks[2].fit

        # only one line found: draw the other one parallel
        if states[1] == states[2] == "measured":
            width = ll.poly_x(right, self.h - 1) - ll.poly_x(left, self.h - 1)
            self.lane_px = (1 - WIDTH_LEARN) * self.lane_px + WIDTH_LEARN * width
        lane_px = self.lane_px
        if left is None and right is not None:
            left, states[1] = right - np.array([0, 0, lane_px]), "inferred"
        elif right is None and left is not None:
            right, states[2] = left + np.array([0, 0, lane_px]), "inferred"

        result = FrameResult(image=undist.copy(), undistorted=undist, warped=warped, binary=binary,
                             hist=hist, calculated_peaks=calculated, sorted_peaks=sorted_peaks,
                             measurements=measured, mode=mode, states=states, roi=self.perspective.roi, accepted=valid,
                             confidences=[t.confidence for t in self.tracks],
                             fits=[t.fit for t in self.tracks])
        if left is None:
            result.status = "missing"
            self.alert_count = 0
            s = self.w / 1280
            cv2.putText(result.image, "No reliable lane estimate", (int(20 * s), int(50 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2 * s, (0, 165, 255), max(2, int(3 * s)), cv2.LINE_AA)
            return result

        ego = states[1:3]
        result.status = ("inferred" if "inferred" in ego else "held" if "held" in ego else "measured")
        result.fits[1], result.fits[2] = left, right
        result.radius_m = float(np.mean([curvature_radius(f, self.w, self.h, self.profile)
                                         for f in (left, right)]))
        result.offset_m = vehicle_offset(left, right, self.perspective.vehicle_x(), self.w, self.h)
        result.width_m = float(lane_width(left, right, self.w, self.h))
        result.alert = self._departure(result)

        neighbours = []
        if result.status == "measured" and SHOW_NEIGHBOURS.get(self.profile, True):
            for i, ref in ((0, left), (3, right)):
                f = self.tracks[i].fit
                if f is not None and states[i] in ("measured", "held"):
                    pair = (f, ref) if i == 0 else (ref, f)
                    if self.plausible_pair(*pair):
                        neighbours.append(f)
        # draw only up to where the lines were really seen
        seen = [t.y_top for t, st in zip(self.tracks[1:3], states[1:3]) if st != "inferred" and t.x is not None]
        result.y_top = self._valid_top(left, right, int(max(seen)) if seen else 0)
        result.image = self.draw_lane(undist, left, right, neighbours, result)
        return result

    def _left_line_not_left_of_yellow(self, measured, valid, binary, yellow):
        """The left line of our lane cannot be left of the yellow line."""
        hist = np.convolve(np.sum(yellow[self.h // 2:] > 0, axis=0).astype(float),
                           np.ones(15) / 15, mode="same")
        x0 = int(np.argmax(hist))
        if hist[x0] < ll.MIN_PEAK or not 0.15 * self.w < x0 < 0.60 * self.w:
            return
        yl = ll.sliding_window_polyfit(yellow, x0)
        if not yl.found:
            return
        ys = np.linspace(yl.ys.min(), self.h - 1, 9)
        x_yellow = ll.poly_x(yl.fit, ys)
        margin = 0.08 * self.w

        def left_of_yellow(fit):
            return fit is not None and np.median(x_yellow - ll.poly_x(fit, ys)) > margin

        # the right line must be clearly right of the yellow line
        if valid[2] and np.median(ll.poly_x(measured[2].fit, ys) - x_yellow) < 0.5 * LANE_WIDTH_REL * self.w:
            valid[2] = False

        track = self.tracks[1]
        cand_wrong = valid[1] and left_of_yellow(measured[1].fit)
        track_wrong = left_of_yellow(track.fit)
        if not (cand_wrong or track_wrong):
            return
        if track_wrong:
            track.reset()
        if self._plausible_line(1, yl):
            measured[1], valid[1] = yl, True
        elif cand_wrong:
            valid[1] = False

    def check_candidates(self, measured, valid, binary, yellow):
        """Rules that need several lines at once."""
        self._left_line_not_left_of_yellow(measured, valid, binary, yellow)
        if (LEFT_LINE_MUST_BE_YELLOW.get(self.profile, False) and valid[1]
                and not self._is_yellow(measured[1].fit, binary, yellow)):
            valid[1] = False
        # yellow is always on the left of our direction: never a right line
        for i in (2, 3):
            if valid[i] and self._is_yellow(measured[i].fit, binary, yellow):
                valid[i] = False
        if valid[1] and valid[2]:
            self._check_width(measured, valid)
        if valid[1] and valid[2]:
            self._make_parallel(measured, valid)
        self._reject_false_neighbours(measured, valid, binary, yellow)
        # neighbouring lines take the shape of the best line of our lane
        ego = [i for i in (1, 2) if valid[i]]
        if ego:
            strong = max(ego, key=lambda i: measured[i].confidence)
            for i in (0, 3):
                if valid[i] and measured[i].confidence < measured[strong].confidence:
                    self._reshape(measured, valid, i, strong)

    def _check_width(self, measured, valid):
        """If the lane width suddenly changes, drop the line that moved."""
        if self.tracks[1].x is None or self.tracks[2].x is None:
            return
        width = ll.poly_x(measured[2].fit, self.h - 1) - ll.poly_x(measured[1].fit, self.h - 1)
        if abs(width - self.lane_px) > MAX_WIDTH_DEVIATION * self.lane_px:
            moved = [self.tracks[i].distance(measured[i].fit) for i in (1, 2)]
            valid[1 if moved[0] > moved[1] else 2] = False

    def _valid_top(self, left, right, y_top):
        """Highest row from which the drawn lane still has a realistic width."""
        c = PAIR_CHECK.get(self.profile, PAIR_CHECK["default"])
        ys = np.arange(self.h - 1, y_top - 1, -4, dtype=float)
        widths = np.array([lane_width(left, right, self.w, self.h, y) for y in ys])
        bad = np.flatnonzero((widths < c["min_m"] - 0.5) | (widths > c["max_m"] + 0.5))
        return int(y_top if len(bad) == 0 else ys[bad[0]] + 1)

    def _make_parallel(self, measured, valid):
        """A weak line takes the shape of the strong line (lane lines are parallel)."""
        weak, strong = (1, 2) if measured[1].confidence < measured[2].confidence else (2, 1)
        if measured[weak].confidence < WEAK_LINE and measured[strong].confidence >= STRONG_LINE:
            self._reshape(measured, valid, weak, strong)

    def _reshape(self, measured, valid, weak, strong):
        """Give line `weak` the shape of line `strong`, keep its own position."""
        w_line, s_line = measured[weak], measured[strong]
        if not len(w_line.xs):
            return
        a, b, _ = s_line.fit
        c = float(np.median(w_line.xs - a * w_line.ys ** 2 - b * w_line.ys))
        w_line.fit = np.array([a, b, c])
        inliers = np.abs(ll.poly_x(w_line.fit, w_line.ys) - w_line.xs) < ll.INLIER_TOLERANCE * self.w
        w_line.xs, w_line.ys = w_line.xs[inliers], w_line.ys[inliers]
        w_line.confidence = ll.coverage(w_line.ys, self.h)
        if not w_line.found:
            valid[weak] = False

    def _reject_false_neighbours(self, measured, valid, binary, yellow):
        """No neighbouring lane behind a yellow or a solid line of our lane."""
        for i, ego in ((0, 1), (3, 2)):
            track = self.tracks[ego]
            if valid[ego]:
                is_solid = float(measured[ego].confidence >= SOLID_MIN)
                track.solid = is_solid if track.x is None else 0.8 * track.solid + 0.2 * is_solid
            if not valid[i]:
                continue
            ego_fit = measured[ego].fit if valid[ego] else track.fit
            if (ego_fit is None or track.solid > 0.5 or
                    (i == 0 and self._is_yellow(ego_fit, binary, yellow))):
                valid[i] = False

    def _is_yellow(self, fit, binary, yellow):
        ys, xs = binary.nonzero()
        near = np.abs(xs - ll.poly_x(fit, ys)) < 0.03 * self.w
        if near.sum() < 50:
            return False
        return bool(np.mean(yellow[ys[near], xs[near]] > 0) > 0.5)

    def _departure(self, r):
        """Warn when the edge of the car comes close to a lane line."""
        near = (r.status == "measured" and
                r.width_m / 2 - abs(r.offset_m) - HALF_CAR_M < DEPARTURE_MARGIN_M)
        self.alert_count = self.alert_count + 1 if near else 0
        return self.alert_count >= DEPARTURE_FRAMES


    def draw_lane(self, undist, left, right, neighbours, r):
        """Lane drawn in the top view and warped back onto the image with Minv."""
        ys = np.arange(r.y_top, self.h, dtype=float)
        layer = np.zeros_like(undist)
        if len(ys) < 2:
            out = undist.copy()
            self.draw_text(out, r)
            return out

        def pts(fit):
            return np.stack([ll.poly_x(fit, ys), ys], axis=1)

        color = (0, 0, 255) if r.alert else (0, 255, 0) if r.status == "measured" else (0, 165, 255)
        cv2.fillPoly(layer, [np.vstack([pts(left), pts(right)[::-1]]).astype(np.int32)], color)
        for nfit in neighbours:
            ref = left if ll.poly_x(nfit, self.h - 1) < ll.poly_x(left, self.h - 1) else right
            cv2.fillPoly(layer, [np.vstack([pts(nfit), pts(ref)[::-1]]).astype(np.int32)], (0, 0, 255))
        for fit in [left, right, *neighbours]:
            cv2.polylines(layer, [pts(fit).astype(np.int32)], False, (0, 255, 255), int(0.012 * self.w))

        out = cv2.addWeighted(undist, 1.0, self.perspective.unwarp(layer), 0.4, 0)
        self.draw_text(out, r)
        return out

    def draw_text(self, img, r):
        s = self.w / 1280
        radius = "> 3000 m (straight)" if r.radius_m > 3000 else f"{r.radius_m:.0f} m"
        side = "right" if r.offset_m > 0 else "left"
        lines = [f"Radius of curvature: {radius}",
                 f"Vehicle is {abs(r.offset_m):.2f} m {side} of center",
                 "Confidence: " + "  ".join(f"L{i} {c * 100:.0f}%" for i, c in enumerate(r.confidences))
                 + f"   [{r.status}]"]
        shade = img.copy()
        cv2.rectangle(shade, (0, 0), (self.w, int(140 * s)), (0, 0, 0), -1)
        cv2.addWeighted(shade, 0.45, img, 0.55, 0, img)
        for i, text in enumerate(lines):
            cv2.putText(img, text, (int(20 * s), int((45 + 40 * i) * s)), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0 * s, (255, 255, 255), max(1, int(2 * s)), cv2.LINE_AA)
        if r.alert:
            cv2.putText(img, "Alert! Lane departure!", (int(0.27 * self.w), int(0.45 * self.h)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6 * s, (0, 0, 255), max(2, int(4 * s)), cv2.LINE_AA)
