import cv2
import numpy as np
import math
import decimal

#gia thiet anh chup duoc la anh vuong de? de~ mo phong?
#input anh va resize ve kich thuoc a*a pixel
a=800
img= cv2.imread("/home/kiencate/Documents/LearnCodePython/learn opencv/img_test.png")

img= cv2.resize(img, (800,800))
cv2.line(img,(0,400),(800,400),(0,255,0),2)
cv2.line(img,(400,0),(400,800),(0,255,0),2)

#chon camera co goc quan sat ngang= doc = 60 do (thong so nay tuy thuoc vao tieu cu, kich thuoc sensor)
h_cam = 100             # chieu cao cua camera so voi mat dat


def dectect_coordinate(x,y):
    cam_coordinate = (0 , 0)
    w_img_met = h_img_met = h_cam * math.tan(60/360 * math.pi) *2 
    x_obj_met = (x - a/2) / a * w_img_met
    y_obj_met = (a/2 - y) / a * h_img_met

    obj_coordinate = ( round( (x_obj_met * 0.000008985 + cam_coordinate[0]), 9)  ,  round( (y_obj_met * 0.000009044+cam_coordinate[1] ), 9)) 
    print(obj_coordinate)


cv2.imshow("coordinate", img)
cv2.waitKey(0)