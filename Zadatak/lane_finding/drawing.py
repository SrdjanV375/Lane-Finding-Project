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


def grid(images, titles, cols, width=640):
    """Images in a grid with `cols` columns (all are resized to the same size)."""
    height = int(width * 9 / 16)
    tiles = [label(cv2.resize(to_bgr(img), (width, height)), t, scale=1.4, bottom=True)
             for img, t in zip(images, titles)]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)
