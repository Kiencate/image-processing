import cv2
from cv2 import Canny
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("Lena.png", 0)
canny = cv2.Canny(img, 100, 200)
sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)
sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))

sobelCombined = cv2.bitwise_or(sobelX, sobelY)

title = ['original image', 'canny', 'sobelX', 'sobelY', 'sobelCombined']
images = [img, canny, sobelX, sobelY, sobelCombined]
for i in range(5):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(title[i])
    plt.xticks([]), plt.yticks([])

plt.show()
