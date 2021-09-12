import cv2
import numpy as np
import math

# h_cam : chieu cao camera
# h,w_corner : goc mo cua camera theo huong doc va ngang
# h,w_img : kich co anh chup
h_cam = 100
h_corner = w_corner = 60
h_img = w_img = 800

imgTest = "/home/kiencate/Documents/LearnCodePython/learn opencv/Screenshot from 2021-09-12 14-46-36.png"
#xac dinh vat the
def getContours(img):
    contours,hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    c=1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if area>300:
            peri = cv2.arcLength(cnt,True)           
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)       
            x,y,w,h = cv2.boundingRect(approx)
            a = x+w/2
            b = y+h/2
            cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(imgContour,str(c),(int(a),int(b)),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,255))
            dectect_coordinate(a,b,c)
            c+=1

#xac dinh toa do vat the
def dectect_coordinate(x,y,c):
    cam_coordinate = (89.999999 , -179.9999)
    w_img_met = h_cam * math.tan(w_corner/180 * math.pi) *2 
    h_img_met = h_cam * math.tan(h_corner/180 * math.pi) *2 
    x_obj_met = (x - w_img/2) / w_img * w_img_met
    y_obj_met = (h_img/2 - y) / h_img * h_img_met

    longtitude = x_obj_met * 0.000008985 + cam_coordinate[1]
    if longtitude >= 180:
        longtitude -= 360
    if longtitude <= -180:
        longtitude += 360

    latitude = y_obj_met * 0.000009044+cam_coordinate[0]
    if latitude >= 90:
        latitude -= 180
    if latitude <= -90:
        latitude += 180
    obj_coordinate = (latitude, longtitude)
    print(c,":",obj_coordinate)


while True:

    img=cv2.imread(imgTest)
    img= cv2.resize(img, (h_img,w_img))
    imgContour = img.copy()
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lower = np.array([11,27,194])
    upper = np.array([43,46,244])
    mask = cv2.inRange(imgHSV,lower,upper)
    getContours(mask)

    cv2.imshow("a",imgContour)
    k= cv2.waitKey(0)
    if k == 27:
        break


