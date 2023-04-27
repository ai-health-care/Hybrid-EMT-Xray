import cv2

import numpy as np

from . import Subsystem


def search_capture(default):
    min_sum = np.inf
    index = default
    for i in range(10):
        cap = cv2.VideoCapture(i) 
        if cap is None or not cap.isOpened():
            continue
        _, image = cap.read()
        cap.release()
        # look for "most grayscaley" image (otherwise cv2 may select laptop webcam)
        # --> also, most likely framegrabber will output a black image first
        s = np.sum(image[..., 0] - image[..., 1])
        if s < min_sum:
            min_sum = s
            index = i
    return index


class FrameGrabber(Subsystem):
    def __init__(self, default_device=2):
        self.capture = cv2.VideoCapture(search_capture(default_device))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 576)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    def available(self):
        return self.capture is not None

    def shoot_image(self, num_frames=100):
        # we'll receive some black frames in the beginning
        while True:
            _, image = self.capture.read()
            if np.sum(image) / (image.shape[0] * image.shape[1]) > 20:
                break
        # average over frames to eliminate noise
        frames = [
            self.capture.read()[1] for _ in range(num_frames * 2)
        ]
        # eliminate ghost images (occur to weird buffering?)
        return np.mean(frames[num_frames:], axis=0)

    def __del__(self):
        self.capture.release()


def crop_xray(xray, bounds=None, autocrop_max=20):
    """
    :param bounds: Attept autocrop if None.
    """
    if bounds is None:
        gray = cv2.cvtColor(xray, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, autocrop_max, 255, cv2.THRESH_BINARY)
        bounds = cv2.boundingRect(thresh)
    x, y, w, h = bounds
    xray_cropped = xray[y:y + h, x:x + w, ...]
    return xray_cropped, bounds