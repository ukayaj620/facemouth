import cv2
import numpy as np
from matplotlib import pyplot as plt


class MouthExtractor:

    def __init__(self):
        self.image_bgr = None
        self.image_hsv = None
        self.face_cascade = cv2.CascadeClassifier(
            './haar_features/haarcascade_frontalface_alt.xml')
        self.mouth_cascade = cv2.CascadeClassifier(
            './haar_features/haarcascade_mcs_mouth.xml')

    def load_image(self, image_path):
        self.image_bgr = cv2.imread(image_path)
        self.image_hsv = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2HSV)
        self.image_gray = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2GRAY)

    def get_face_roi(self, image_gray):
        return self.face_cascade.detectMultiScale(image_gray, scaleFactor=1.05,
                                                  minNeighbors=5, minSize=(500, 500),
                                                  flags=cv2.CASCADE_SCALE_IMAGE)

    def get_mouth_roi(self, face_image_gray, face_rect):
        (fx, fy, fw, fh) = face_rect
        face_image_gray_cropped = face_image_gray[fy +
                                                  int(fh * 0.5):fy + fh, fx:fx + fw]
        return self.mouth_cascade.detectMultiScale(
            face_image_gray_cropped, scaleFactor=1.1, minNeighbors=10, minSize=(300, 200), flags=cv2.CASCADE_SCALE_IMAGE)

    def draw_mouth_boundaries(self, image, face_rect, mouth_rects):
        sorted_mouth_rect = sorted(mouth_rects, key=lambda rect: rect[1], reverse=True)
        (fx, fy, fw, fh) = face_rect
        (mx, my, mw, mh) = sorted_mouth_rect[0]

        cv2.rectangle(image, (fx + mx, fy + int(fh * 0.5) + my),
                      (fx + mx + mw, fy + int(fh * 0.5) + my + mh), (255, 0, 0), thickness=12)

    def draw_face_boundaries(self, image, face_rect):
        (fx, fy, fw, fh) = face_rect

        cv2.rectangle(image, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), thickness=12)

    def get_mouth_boundaries(self):
        image = self.image_bgr.copy()
        face_rects = self.get_face_roi(self.image_gray)

        if len(face_rects) == 0:
          return image

        face_rect = face_rects[0]
        mouth_rects = self.get_mouth_roi(self.image_gray, face_rect)

        if len(mouth_rects) == 0:
            return image

        self.draw_mouth_boundaries(image, face_rect, mouth_rects)
        self.draw_face_boundaries(image, face_rect)
        
        return image
