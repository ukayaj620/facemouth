import unittest
import matplotlib.pyplot as plt
import cv2

from facial_extractor.mouth_extractor import MouthExtractor


class TestMouthExtractor(unittest.TestCase):

    def setUp(self):
        self.mouth_extractor = MouthExtractor()

    def test_load_image(self):
        self.mouth_extractor.load_image("./data/jayaku/foto_1.jpg")
        plt.imshow(self.mouth_extractor.image_bgr[..., ::-1])
        plt.show()
        plt.imshow(self.mouth_extractor.image_hsv)
        plt.show()

    def test_mouth_boundaries(self):
        names = ["ferdy", "jason", "jayaku", "kevin", "martien"]
        fig = plt.figure(figsize=(10, 12))
        columns = 5
        rows = 6
        for i in range(0, columns * rows):
            name = names[(i % columns)]
            self.mouth_extractor.load_image(
                "./data/{}/foto_{}.jpg".format(name, (i % rows + 1)))
            image = self.mouth_extractor.get_mouth_boundaries()[..., ::-1]
            fig.add_subplot(rows, columns, i + 1)
            plt.imshow(cv2.resize(image, (300, 400)))

        plt.show()

    def test_mouth_boundaries_personal(self):
        name = "kevin"
        fig = plt.figure(figsize=(10, 12))
        columns = 3
        rows = 2
        for i in range(1, columns * rows + 1):
            self.mouth_extractor.load_image(
                "./data/{}/foto_{}.jpg".format(name, i))
            image = self.mouth_extractor.get_mouth_boundaries()[..., ::-1]
            fig.add_subplot(rows, columns, i)
            plt.imshow(cv2.resize(image, (300, 400)))

        plt.show()