import cv2
from matplotlib import pyplot as plt
image = cv2.imread("images.jpeg",0)
# image = cv2.resize(image,(256,256))

_,image_thres_global = cv2.threshold(image,50,255,cv2.THRESH_BINARY)
_,image_thres_otsu = cv2.threshold(image,120,255,cv2.THRESH_OTSU)
image_thres_adaptive =cv2.adaptiveThreshold(image, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
title = ['original image', 'global threshold', 'otsu threshold', 'localy adaptive threshold']
images = [image,image_thres_global,image_thres_otsu,image_thres_adaptive]
for i in range(4):
    plt.subplot(2, 2, i+1), plt.imshow(cv2.cvtColor(images[i],cv2.COLOR_BGR2RGB))
    plt.title(title[i])
    plt.xticks([]), plt.yticks([])

plt.show()