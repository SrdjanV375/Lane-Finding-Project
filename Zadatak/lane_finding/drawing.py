"""Helpers for building figures for the writeup."""
import cv2
import numpy as np


def to_bgr(img):
    """Binary/gray image -> 3-channel image (so it can be stacked with color images)."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def label(img, text, color=(0, 255, 0), scale=1.0, bottom=False):
    img = to_bgr(img).copy()
    s = img.shape[1] / 1280 * scale
    y = img.shape[0] - int(20 * s) if bottom else int(50 * s)
    cv2.putText(img, text, (int(20 * s), y), cv2.FONT_HERSHEY_SIMPLEX, 1.4 * s,
                color, max(1, int(3 * s)), cv2.LINE_AA)
    return img


def side_by_side(images, titles, width=640):
    """Resize images to the same width, put a title on each and stack them horizontally."""
    out = []
    for img, title in zip(images, titles):
        img = to_bgr(img)
        img = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])))
        out.append(label(img, title, (0, 0, 255), scale=2.0))
    return np.hstack(out)
