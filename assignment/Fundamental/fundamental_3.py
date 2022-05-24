import numpy as np
import cv2
from matplotlib import pyplot as plt
def change_contrast(img,alpha = 1):
    min = np.min(img)
    max = np.max(img)
    min_ = int((max+min)/2 - (max-min)*alpha/2)
    max_ = int((max+min)/2 + (max-min)*alpha/2)
    a = (max_-min_)/(max-min)
    b = (min_*max - min*max_)/(max-min)

    img_ = np.zeros_like(img,np.uint8)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_[i][j] = int(np.clip(img[i][j]*a + b,0,255))
    return img_

img = cv2.imread("/home/kiencate/Pictures/under_water.png",0)
img_equal = cv2.equalizeHist(img)
img_change_contrast_opencv = cv2.convertScaleAbs(img, alpha=2, beta=0)
img_change_contrast = change_contrast(img,alpha=5)
result = np.hstack((img,img_equal,img_change_contrast))
compare = np.hstack((img_change_contrast,img_change_contrast_opencv))
cv2.imshow("change contrast code\t\t\t\t change contrast opencv",compare)
cv2.imshow("original\t\t\t\t\t\t equalization\t\t\t\t\t\t\t contrast", result)

img_hist = cv2.calcHist([img],[0],None,[256],[0,256])
img_equal_hist = cv2.calcHist([img_equal],[0],None,[256],[0,256])
img_codan_hist = cv2.calcHist([img_change_contrast],[0],None,[256],[0,256])
plt.subplot(131),plt.plot(img_hist),plt.title("original")
plt.subplot(132),plt.plot(img_equal_hist),plt.title("equalization")
plt.subplot(133),plt.plot(img_codan_hist),plt.title("change contrast")
plt.show()

cv2.imwrite("equalization_3.jpg",img_equal)
cv2.imwrite("change_contrast_3.jpg",img_change_contrast)