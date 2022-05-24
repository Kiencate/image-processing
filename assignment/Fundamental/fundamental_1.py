import cv2
img = cv2.imread("Lena.png")
cv2.imshow("lena", img)
cv2.waitKey(0)
cv2.imwrite("Lena1.png", img)
