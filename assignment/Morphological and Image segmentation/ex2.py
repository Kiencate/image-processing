import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("man.jpg",0)

_,image = cv2.threshold(image,0,255,cv2.THRESH_OTSU)
kernel = np.ones((9,9),np.uint8)
image_erosion = cv2.erode(image,kernel)
image_dialation = cv2.dilate(image,kernel)
image_open = cv2.morphologyEx(image,cv2.MORPH_OPEN,kernel)
image_close = cv2.morphologyEx(image,cv2.MORPH_CLOSE,kernel)

title = ['original image', 'image erosion', 'image dialation', 'image open','image close']
images = [image,image_erosion,image_dialation,image_open,image_close]
for i in range(5):
    plt.subplot(2, 3, i+1), plt.imshow(cv2.cvtColor(images[i],cv2.COLOR_BGR2RGB))
    plt.title(title[i])
    plt.xticks([]), plt.yticks([])

plt.show()