import unittest
import matplotlib.pyplot as plt
import cv2

from facial_extractor.face_extractor import FaceExtractor


class TestFaceExtractor(unittest.TestCase):

    def setUp(self):
        self.face_extractor = FaceExtractor()

    def test_load_image(self):
        self.face_extractor.load_image("./data/jayaku/foto_1.jpg")
        plt.imshow(self.face_extractor.image_bgr[..., ::-1])
        plt.show()
        plt.imshow(self.face_extractor.image_hsv)
        plt.show()

    def test_face_mask(self):
        self.face_extractor.load_image("./data/jayaku/foto_1.jpg")
        plt.imshow(self.face_extractor.get_face_mask())
        plt.show()
    
    def test_face_mask(self):
        names = ["ferdy", "jason", "jayaku", "kevin", "martien"]
        fig = plt.figure(figsize=(10, 12))
        columns = 5
        rows = 6
        for i in range(0, columns * rows):
            name = names[(i % columns)]
            self.face_extractor.load_image(
                "./data/{}/foto_{}.jpg".format(name, (i % rows + 1)))
            image = self.face_extractor.get_face_mask()
            fig.add_subplot(rows, columns, i + 1)
            plt.imshow(cv2.resize(image, (300, 400)), cmap="gray")

        plt.show()

    def test_apply_mask(self):
        names = ["ferdy", "jason", "jayaku", "kevin", "martien"]
        fig = plt.figure(figsize=(10, 12))
        columns = 5
        rows = 6
        for i in range(0, columns * rows):
            name = names[(i % columns)]
            self.face_extractor.load_image(
                "./data/{}/foto_{}.jpg".format(name, (i % rows + 1)))
            image = self.face_extractor.apply_mask(
                self.face_extractor.get_face_mask())[..., ::-1]
            fig.add_subplot(rows, columns, i + 1)
            plt.imshow(cv2.resize(image, (300, 400)))

        plt.show()

    def test_face_contour(self):
        names = ["ferdy", "jason", "jayaku", "kevin", "martien"]
        fig = plt.figure(figsize=(10, 12))
        columns = 5
        rows = 6
        for i in range(0, columns * rows):
            name = names[(i % columns)]
            self.face_extractor.load_image(
                "./data/{}/foto_{}.jpg".format(name, (i % rows + 1)))
            image = self.face_extractor.get_face_contour()[..., ::-1]
            fig.add_subplot(rows, columns, i + 1)
            plt.imshow(cv2.resize(image, (300, 400)))
        
        plt.show()
    
    def test_face_mask_personal(self):
        name = "kevin"
        fig = plt.figure(figsize=(10, 12))
        columns = 3
        rows = 2
        for i in range(1, columns * rows + 1):
            self.face_extractor.load_image(
                "./data/{}/foto_{}.jpg".format(name, i))
            image = self.face_extractor.get_face_mask()
            fig.add_subplot(rows, columns, i)
            plt.imshow(cv2.resize(image, (300, 400)), cmap="gray")

        plt.show()

    def test_apply_mask_personal(self):
        name = "martien"
        fig = plt.figure(figsize=(10, 12))
        columns = 3
        rows = 2
        for i in range(1, columns * rows + 1):
            self.face_extractor.load_image(
                "./data/{}/foto_{}.jpg".format(name, i))
            image = self.face_extractor.apply_mask(
                self.face_extractor.get_face_mask())[..., ::-1]
            fig.add_subplot(rows, columns, i)
            plt.imshow(cv2.resize(image, (300, 400)))

        plt.show()

    def test_face_contour_personal(self):
        name = "martien"
        fig = plt.figure(figsize=(10, 12))
        columns = 3
        rows = 2
        for i in range(1, columns * rows + 1):
            self.face_extractor.load_image(
                "./data/{}/foto_{}.jpg".format(name, i))
            image = self.face_extractor.get_face_contour()[..., ::-1]
            fig.add_subplot(rows, columns, i)
            plt.imshow(cv2.resize(image, (300, 400)))

        plt.show()


if __name__ == "__main__":
    unittest.main()
