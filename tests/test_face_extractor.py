import unittest
import matplotlib.pyplot as plt

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

    def test_apply_mask(self):
        self.face_extractor.load_image("./data/jason/foto_1.jpg")
        image_result = self.face_extractor.apply_mask(
            self.face_extractor.get_face_mask())

        plt.imshow(image_result[..., ::-1])
        plt.show()


if __name__ == "__main__":
    unittest.main()
