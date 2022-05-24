import cv2
from cv2 import calcHist
import numpy as np
from matplotlib import pyplot as plt

def hist_dev(img,bin):
    hist = np.zeros((bin,1),dtype=int)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist[int(img[i][j]/255*bin)] +=1
    return hist
    
img = cv2.imread("Lena.png",0)

my_hist = hist_dev(img,256)
cv2_hist = cv2.calcHist([img],[0],None,[256],[0,256])

plt.subplot(121),plt.plot(cv2_hist),plt.title("hist of cv2")
plt.subplot(122),plt.plot(my_hist),plt.title("my hist")
plt.show()
