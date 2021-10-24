import cv2
import numpy as np


class FaceExtractor:

    def __init__(self):
        self.image_bgr = None
        self.image_hsv = None

    def load_image(self, image_path):
        self.image_bgr = cv2.imread(image_path)
        self.image_hsv = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2HSV)

    def create_kernel(self, size):
        return np.ones((size, size), dtype=np.uint8)

    def create_hsv_color_mask(self, lower, upper):
        return cv2.inRange(self.image_hsv, np.array(lower, dtype="uint8"), np.array(upper, dtype="uint8"))

    def apply_opening(self, mask, kernel):
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    def apply_closing(self, mask, kernel):
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    def get_face_mask(self):
        mask = self.create_hsv_color_mask([0, 48, 15], [20, 255, 255])
        mask_open = self.apply_opening(mask, self.create_kernel(size=9))
        return self.apply_closing(mask_open, self.create_kernel(size=199))

    def apply_mask(self, mask):
        return cv2.bitwise_and(self.image_bgr, self.image_bgr, mask=mask)

    def get_face(self):
        mask = self.get_face_mask()
        return self.apply_mask(mask)

    def get_face_contour(self):
        mask = self.get_face_mask()
        contours, hierarchy = cv2.findContours(
            image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        image = self.image_bgr.copy()
        cv2.drawContours(image=image, contours=contours, contourIdx=-1,
                         color=(0, 255, 0), thickness=8, lineType=cv2.LINE_AA)
        return image
