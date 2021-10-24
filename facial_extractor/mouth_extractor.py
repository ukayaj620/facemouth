import cv2
import numpy as np
from matplotlib import pyplot as plt


class MouthExtractor:

    def __init__(self):
        self.image_bgr = None
        self.image_hsv = None
        self.image_gray = None
        self.face_cascade = cv2.CascadeClassifier(
            './../haar_features/haarcascade_frontalface_alt.xml')
        self.mouth_cascade = cv2.CascadeClassifier(
            './../haar_features/haarcascade_mcs_mouth.xml')

    def load_image(self, image_path):
        self.image_bgr = cv2.imread(image_path)
        self.image_hsv = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2HSV)
        self.image_gray = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2GRAY)

    def create_kernel(self, size):
        return np.ones((size, size), dtype=np.uint8)

    def apply_opening(self, image, kernel):
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    def apply_closing(self, image, kernel):
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    def apply_mask(self, image, mask):
        return cv2.bitwise_and(image, image, mask=mask)

    def create_hsv_color_mask(self, image_hsv, lower, upper):
        return cv2.inRange(image_hsv, np.array(lower, dtype="uint8"), np.array(upper, dtype="uint8"))

    def get_mouth_mask(self, image_hsv):
        mask = self.create_hsv_color_mask(image_hsv, [4, 48, 15], [20, 255, 255])
        mask_closing = self.apply_closing(mask, self.create_kernel(size=9))
        return self.apply_opening(mask_closing, self.create_kernel(size=9))

    def get_image_mouth(self):
        mouth_rects = self.get_mouth_rects()
        if len(mouth_rects) == 0:
            return self.image_bgr
        mouth_rect = self.get_most_bottom_rect(mouth_rects)
        mouth_image = self.get_image_slice_from_rect(self.image_bgr, mouth_rect)

        mouth_image_hsv = cv2.cvtColor(mouth_image, cv2.COLOR_BGR2HSV)
        mouth_mask = self.get_mouth_mask(mouth_image_hsv)

        mouth_mask_invert = cv2.bitwise_not(mouth_mask)

        return self.apply_mask(mouth_image, mouth_mask_invert)


    def _sobel_mouth(self, mouth_image):
        mouth_image_gray = cv2.cvtColor(mouth_image, cv2.COLOR_BGR2GRAY)
        mouth_image_gray_blur = cv2.GaussianBlur(mouth_image_gray, (3, 3), sigmaX=0, sigmaY=0)
        sobelxy = cv2.Sobel(src=mouth_image_gray_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=31)

        magnitude = np.sqrt(np.square(sobelxy))
        magnitude *= 255.0 / magnitude.max()

        return magnitude

    def _mouth_threshold_detection(self, mouth_image):
        mouth_image_gray = cv2.cvtColor(mouth_image, cv2.COLOR_BGR2GRAY)
        retval, mouth_thresh = cv2.threshold(mouth_image_gray, 140, 255, cv2.THRESH_BINARY_INV)

        open_kernel = self.create_kernel(19)
        close_kernel = self.create_kernel(9)
        mouth_thresh = self.apply_opening(mouth_thresh, open_kernel)
        mouth_thresh = self.apply_closing(mouth_thresh, close_kernel)

        return self.apply_mask(mouth_image, mouth_thresh)

    def get_image_with_face_and_mouth_boundaries(self):
        image = self.image_bgr.copy()

        face_rects = self.get_face_rects()
        if len(face_rects) == 0:
            return image
        face_rect = face_rects[0]
        self.draw_face_boundaries(image, face_rect)

        mouth_rects = self.get_mouth_rects(face_rect)
        if len(mouth_rects) == 0:
            return image
        mouth_rect = self.get_most_bottom_rect(mouth_rects)
        self.draw_mouth_boundaries(image, mouth_rect)


        return image

    def get_face_rects(self):
        return self.get_face_roi(self.image_gray)

    def get_mouth_rects(self, face_rect=None):
        if face_rect is None:
            face_rects = self.get_face_rects()

            if len(face_rects) == 0:
                return []

            face_rect = face_rects[0]

        return self.get_mouth_roi(face_rect)

    def get_face_roi(self, image_gray):
        return self.face_cascade.detectMultiScale(
            image_gray, scaleFactor=1.05,
            minNeighbors=5, minSize=(200, 200),
            flags=cv2.CASCADE_SCALE_IMAGE)

    def get_mouth_roi(self, face_rect):
        (fx, fy, fw, fh) = face_rect
        bottom_half_rect = (fx, fy + int(fh * 0.5), fw, int(fh * 0.5))
        face_image_gray_cropped = self.get_image_slice_from_rect(self.image_gray, bottom_half_rect)

        local_mouth_rects = self.mouth_cascade.detectMultiScale(
            face_image_gray_cropped,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        return (bottom_half_rect[0], bottom_half_rect[1], 0, 0) + local_mouth_rects

    def get_image_slice_from_rect(self, image, rect):
        (x, y, w, h) = rect
        return image[y:(y + h), x:(x + w)]

    def get_most_bottom_rect(self, rect):
        return sorted(rect, key=lambda _rect: _rect[1], reverse=True)[0]

    def draw_mouth_boundaries(self, image, mouth_rect):
        self.draw_rect(image, mouth_rect, (255, 0, 0), 12)

    def draw_face_boundaries(self, image, face_rect):
        self.draw_rect(image, face_rect, (0, 255, 0), 12)

    def draw_rect(self, image, rect, color=(0,0,0), thickness=1):
        (x, y, w, h) = rect
        cv2.rectangle(
            image,
            (x, y), (x + w, y + h),
            color,
            thickness=thickness
        )
