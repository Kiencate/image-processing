import cv2
from matplotlib.pyplot import contour

image = cv2.imread("shape.jpg")
image=cv2.resize(image,(800,800))
image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

_,image_bin = cv2.threshold(image_gray,210,255,cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(image_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for cnt in contours:
    print ("ok")
    peri = cv2.arcLength(cnt, True)
    s = cv2.contourArea(cnt)
    approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
    x, y, w, h = cv2.boundingRect(approx)
    cv2.putText(image,"s = %0.0f , p= %0.0f"% (s,peri),(x-15, y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0))
cv2.imshow("result",image)
cv2.waitKey(0)