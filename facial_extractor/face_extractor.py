import cv2
import numpy as np


class FaceExtractor:

    def __init__(self):
        self.image_bgr = None
        self.image_hsv = None

    def load_image(self, image_path):
        self.image_bgr = cv2.imread(image_path)
        self.image_hsv = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2HSV)

    def get_face_mask(self):
        lower = np.array([0, 48, 10], dtype="uint8")
        upper = np.array([20, 255, 255], dtype="uint8")

        mask = cv2.inRange(self.image_hsv, lower, upper)

        kernel_open = np.ones((9, 9), dtype=np.uint8)
        mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

        kernel_close = np.ones((199, 199), dtype=np.uint8)
        return cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel_close)

    def apply_mask(self, mask):
        return cv2.bitwise_and(self.image_bgr, self.image_bgr, mask=mask)

    def get_face(self):
        mask = self.get_face_mask()
        return self.apply_mask(mask)
