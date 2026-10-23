import numpy as np

from .lane_lines import poly_x
from .perspective import LANE_WIDTH_REL

LANE_WIDTH_M = 3.7

# length of road (m) covered by the full height of the bird's-eye view, per camera profile
VIEW_LENGTH_M = {"highway_1280": 20.0, "highway_960": 20.0, "low_camera": 20.0, "mountain": 13.0}


def meters_per_pixel(w, h, profile="highway_1280"):
    xm = LANE_WIDTH_M / (LANE_WIDTH_REL * w)
    ym = VIEW_LENGTH_M.get(profile, 20.0) / h
    return xm, ym


def curvature_radius(fit, w, h, profile="highway_1280"):
    """
    x = A*y^2 + B*y + C in pixels -> A_m = A*xm/ym^2, B_m = B*xm/ym
    R = (1 + (2*A_m*y + B_m)^2)^(3/2) / |2*A_m|
    """
    xm, ym = meters_per_pixel(w, h, profile)
    a = fit[0] * xm / ym ** 2
    b = fit[1] * xm / ym
    y = (h - 1) * ym
    if abs(a) < 1e-9:
        return float("inf")
    return float((1 + (2 * a * y + b) ** 2) ** 1.5 / abs(2 * a))


def lane_width(left_fit, right_fit, w, h, y=None):
    xm, _ = meters_per_pixel(w, h)
    y = h - 1 if y is None else y
    return (poly_x(right_fit, y) - poly_x(left_fit, y)) * xm


def vehicle_offset(left_fit, right_fit, vehicle_x, w, h):
    """Distance of the car from the lane centre (m); positive = car is right of the centre."""
    xm, _ = meters_per_pixel(w, h)
    lane_center = (poly_x(left_fit, h - 1) + poly_x(right_fit, h - 1)) / 2
    return float((vehicle_x - lane_center) * xm)
