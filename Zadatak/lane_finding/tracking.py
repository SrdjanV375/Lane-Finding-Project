"""Kalman filter for one lane line.

The line is kept as its x position at three heights of the top view (top, middle,
bottom). Each frame: predict (the line stays where it was), then correct with the new
measurement - a measurement with little support changes the line less. A line that is
not found is held for max_lost frames and then dropped.
"""
import numpy as np

from .lane_lines import poly_x


class LineKalman:
    def __init__(self, h, process_noise=4.0, measurement_noise=10.0, max_lost=8):
        self.ys = np.array([0.0, h / 2, h - 1.0])
        self.q = process_noise ** 2
        self.r = measurement_noise ** 2
        self.max_lost = max_lost
        self.x = None              # x of the line at self.ys
        self.P = None              # uncertainty (3x3)
        self.lost = 0
        self.confidence = 0.0
        self.y_top = 0.0           # highest row where the line was seen
        self.solid = 0.0           # how often the line looked solid (not dashed)

    def _to_state(self, fit):
        return poly_x(fit, self.ys)

    def _to_fit(self, state):
        return np.polyfit(self.ys, state, 2)

    @property
    def fit(self):
        return None if self.x is None else self._to_fit(self.x)

    @property
    def state(self):
        if self.x is None:
            return "missing"
        return "measured" if self.lost == 0 else "held"

    def predict(self):
        if self.x is not None:
            self.P = self.P + self.q * np.eye(3)

    def update(self, fit, confidence, y_top=0.0):
        z = self._to_state(fit)
        # the top grows slowly, but never above where the line is seen now
        smoothed = y_top if self.x is None else 0.7 * self.y_top + 0.3 * y_top
        self.y_top = max(y_top, smoothed)
        if self.x is None:
            self.x, self.P = z, self.r * np.eye(3)
        else:
            R = self.r / max(confidence, 0.1) * np.eye(3)
            S = self.P + R
            K = self.P @ np.linalg.inv(S)                 # Kalman gain
            self.x = self.x + K @ (z - self.x)
            self.P = (np.eye(3) - K) @ self.P
        self.lost = 0
        self.confidence = confidence

    def miss(self):
        self.lost += 1
        self.confidence = 0.0
        if self.lost > self.max_lost:
            self.reset()

    def reset(self):
        self.x = self.P = None
        self.lost = 0
        self.confidence = 0.0
        self.y_top = 0.0
        self.solid = 0.0

    def distance(self, fit, y_from=0.0):
        """Largest sideways distance (px) between the tracked line and a new one, from y_from down."""
        if self.x is None:
            return 0.0
        ys = np.linspace(min(y_from, self.ys[-1] - 1), self.ys[-1], 5)
        return float(np.max(np.abs(poly_x(fit, ys) - poly_x(self.fit, ys))))
