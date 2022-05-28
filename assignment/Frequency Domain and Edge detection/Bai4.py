import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("Lena.png")


def HSV2RGB(img):
    img_re = np.copy(img)
    for i in img_re:
        for j in i:
            (h, s, v) = j
            hi = math.floor(h / 60.0) % 6
            f = (h / 60.0) - math.floor(h / 60.0)
            p = int(v * (1.0 - s))
            q = int(v * (1.0 - (f*s)))
            t = int(v * (1.0 - ((1.0 - f) * s)))
            j = {
                0: (v, t, p),
                1: (q, v, p),
                2: (p, v, t),
                3: (p, q, v),
                4: (t, p, v),
                5: (v, p, q),
            }[hi]
    return img_re


def RGB2HSV(img):
    img_re = np.copy(img)
    for i in img_re:
        for j in i:
            (r, g, b) = j.astype("int")
            maxc = max(r, g, b)
            minc = min(r, g, b)
            colorMap = {
                id(r): 'r',
                id(g): 'g',
                id(b): 'b'
            }
            if colorMap[id(maxc)] == colorMap[id(minc)]:
                h = 0
            elif colorMap[id(maxc)] == 'r':
                h = 60.0 * ((g - b) / (maxc - minc)) % 360.0
            elif colorMap[id(maxc)] == 'g':
                h = 60.0 * ((b - r) / (maxc - minc)) + 120.0
            elif colorMap[id(maxc)] == 'b':
                h = 60.0 * ((r - g) / (maxc - minc)) + 240.0
            v = maxc
            if maxc == 0.0:
                s = 0.0
            else:
                s = 1.0 - (minc / maxc)
            j = (h, s, v)
    return img_re


def changeHSV(img_HSV, h_change=0, s_change=0, v_change=0):
    # h_change from -360->360, s_change from -1 -> 1, v_change from -255 -> 255
    img_re = np.copy(img_HSV)
    for i in img_re:
        for j in i:
            j[0] = np.clip(j[0]+h_change, 0, 360)
            j[1] = np.clip(j[1]+s_change, 0, 1.0)

            j[2] = np.clip(j[2]+v_change, 0, 255)
    return img_re


imgHSV = changeHSV(RGB2HSV(img), 0, 0, 150)
img_re = HSV2RGB(imgHSV)

result = np.hstack((img,img_re))
cv2.imshow("original image\t\t\t\t\t\t\t\tafter change HSV",result)
cv2.waitKey(0)
