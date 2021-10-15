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
        lower = np.array([0, 48, 80], dtype="uint8")
        upper = np.array([20, 255, 255], dtype="uint8")

        return cv2.inRange(self.image_hsv, lower, upper)

    def apply_mask(self, mask):
        return cv2.bitwise_and(self.image_bgr, self.image_bgr, mask=mask)

    def get_face(self):
        mask = self.get_face_mask()
        return self.apply_mask(mask)
