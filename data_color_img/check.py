import cv2
import numpy as np
img = cv2.imread("ori5.jpg")
img = img[0:800,0:1600]
cv2.imwrite('ori5.jpg',img)