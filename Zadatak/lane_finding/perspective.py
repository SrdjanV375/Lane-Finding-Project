import cv2
import numpy as np

PROFILES_SRC = {
    # test_images/straight_lines1.jpg and straight_lines2.jpg (1280x720, hood at y ~ 0.94h)
    "highway_1280": [(0.443, 0.650), (0.557, 0.650), (0.801, 0.940), (0.199, 0.940)],
    # test_images/solidWhiteRight.jpg and solidYellowLeft.jpg (960x540, no hood visible)
    "highway_960": [(0.428, 0.650), (0.572, 0.650), (0.816, 0.940), (0.184, 0.940)],
    # challenge02.mp4: same resolution but the camera is mounted lower (horizon at ~0.61h);
    # points taken from the straight part of the video (frames 300-480)
    "low_camera": [(0.472, 0.676), (0.568, 0.676), (0.772, 0.940), (0.268, 0.940)],
    # challenge03.mp4: mountain road with sharp curves - shorter look-ahead;
    # points taken from the straight part of the video (frame 559)
    "mountain": [(0.393, 0.700), (0.584, 0.700), (0.781, 0.940), (0.195, 0.940)],
}
DST_REL = [(0.35, 0.0), (0.65, 0.0), (0.65, 1.0), (0.35, 1.0)]

ROI_MARGIN = 0.15
ROI_SIDE_DROP = 0.10

# Lane width in the bird's-eye view, relative to the image width
LANE_WIDTH_REL = DST_REL[1][0] - DST_REL[0][0]

def default_profile(img_size):
    w, _ = img_size
    return "highway_960" if w == 960 else "highway_1280"


class Perspective:

    def __init__(self, img_size, profile=None):
        self.w, self.h = img_size
        self.profile = profile or default_profile(img_size)
        self.src = np.float32([[x * self.w, y * self.h] for x, y in PROFILES_SRC[self.profile]])
        self.dst = np.float32([[x * self.w, y * self.h] for x, y in DST_REL])
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.Minv = cv2.getPerspectiveTransform(self.dst, self.src)
        self.roi_polygon = self._roi_polygon()
        self.roi = self.warp(self._roi_camera(), flags=cv2.INTER_NEAREST)

    def _roi_polygon(self):

        tl, tr, br, bl = self.src
        m = ROI_MARGIN * self.w
        side_y = tl[1] + ROI_SIDE_DROP * self.h
        return np.int32([[tl[0] - m, tl[1]], [tr[0] + m, tr[1]], [self.w - 1, side_y],
                         [self.w - 1, br[1]], [0, bl[1]], [0, side_y]])

    def _roi_camera(self):
        mask = np.zeros((self.h, self.w), np.uint8)
        cv2.fillPoly(mask, [self.roi_polygon], 255)
        return mask

    def warp(self, img, flags=cv2.INTER_LINEAR):
        return cv2.warpPerspective(img, self.M, (self.w, self.h), flags=flags)

    def vehicle_x(self):
        bottom_y = self.src[2][1]
        pt = cv2.perspectiveTransform(np.float32([[[self.w / 2, bottom_y]]]), self.M)
        return float(pt[0, 0, 0])

    def unwarp(self, img, flags=cv2.INTER_LINEAR):
        return cv2.warpPerspective(img, self.Minv, (self.w, self.h), flags=flags)


def warper(img, src, dst):

    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)

    return warped
