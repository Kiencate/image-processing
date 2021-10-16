import cv2
import numpy as np
import math
import random
size =200

m = 0
for a in range(1):
    n=0
    img = cv2.imread(f'ori7.jpg')
    img_crop = np.zeros((1000,size,size,3), np.uint8)
    for x in range(500):
        i= random.randint(238,670)
        j = random.randint(0,670)
        img_crop[n] = img[i:i+200, j:j+200]      
        cv2.imwrite(f'{m}.jpg', img_crop[n])       
        n+=1
        m+=1
         
cv2.waitKey(0)
