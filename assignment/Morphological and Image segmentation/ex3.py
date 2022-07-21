
import cv2

import numpy as np
from matplotlib import pyplot as plt
 

img = cv2.imread('connected.png')
gray_img = cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)
 
blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)
 
threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
image_canny = cv2.Canny(threshold,5,200)
# Apply the Component analysis function
analysis = cv2.connectedComponentsWithStats(threshold,
                                            4,
                                            cv2.CV_32S)
(totalLabels, label_ids, values, centroid) = analysis
print("number of connected set: ",totalLabels)

output = np.zeros(gray_img.shape, dtype="uint8")
 
title = ['original image', f'image binary, connected set = {totalLabels}', 'image canny']
images = [img,threshold,image_canny]
for i in range(3):
    plt.subplot(3, 1, i+1), plt.imshow(cv2.cvtColor(images[i],cv2.COLOR_BGR2RGB))
    plt.title(title[i])
    plt.xticks([]), plt.yticks([])

plt.show()