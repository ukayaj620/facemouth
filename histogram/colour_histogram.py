import cv2
import numpy
import matplotlib.pyplot as plt

img = cv2.imread("../tests/data/jason/foto_5.jpg")

plt.figure()

channels = ['b', 'g', 'r']
for idx, channel in enumerate(channels):
    hist = cv2.calcHist([img], [idx], None, [256], [0, 256])
    hist = hist/hist.sum()

    plt.title("RGB Histogram (Normalized)")
    plt.xlabel("Bins")
    plt.ylabel("% of Pixels")
    plt.plot(hist, color=channel)
    plt.xlim([0, 256])

plt.show()
