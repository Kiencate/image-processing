
import cv2
import numpy as np
print(cv2.__version__)
kernel = np.ones((5,5),np.uint8)
avt = "/home/kiencate/Documents/LearnCodePython/learn opencv/avt.jpg"
card = "/home/kiencate/Documents/LearnCodePython/learn opencv/card.jpg"
car = "/home/kiencate/Documents/LearnCodePython/learn opencv/car.jpg"
shape= "/home/kiencate/Documents/LearnCodePython/learn opencv/shape.png"
#su dung cam truoc
cap = cv2.VideoCapture(0)
# cap.set(3,640)
# cap.set(4,640)
# print(cap)
# while True:
#     success, img = cap.read()
#     cv2.imshow("Videohai",img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

#thay doi? kich' co~ anh?
# img = cv2.imread(avt)
# (h, w, d) = img.shape
# r = 300.0 / w 
# dim = (int(h * r),300)
# resized = cv2.resize(img, dim)
# cv2.imshow("kiendz",resized)
# cv2.waitKey(0)

#cat anh
# img = cv2.imread("/home/kiencate/Documents/LearnCodePython/learn opencv/avt.jpg")
# crop= img[20:150, 30: 940] #height : width
# cv2.imshow("after crop", crop)
# cv2.waitKey(0)

#hieu ung anh
# img = cv2.imread("/home/kiencate/Documents/LearnCodePython/learn opencv/avt.jpg")
# (h, w, d) = img.shape
# r = 300.0 / w 
# dim = (int(h * r),300)
# img= cv2.resize(img, dim)
# imgGray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# imgBlur = cv2.GaussianBlur(imgGray,(7,7),0)
# imgCanny= cv2.Canny(imgGray,100,100)
# imgDialation= cv2.dilate(img,kernel,iterations=1)
# imgEroded = cv2.erode(imgDialation,kernel, iterations=1)

# cv2.imshow("den trang",imgGray)
# cv2.imshow("am mau",imgBlur)
# cv2.imshow("hoat. hoa.",imgCanny)
# cv2.imshow("gian~ no?",imgDialation)
# cv2.imshow("hoat. hoa. 3",imgEroded)
# cv2.waitKey(0)


# tao anh? va` ve~
# img= np.zeros((512,512,3),np.uint8)
# img[200:300,200:300]=100,0,0               #doi mau anh
# cv2.line(img,(0,0),(300,300),(0,255,0),50)    #ve duong thang
# cv2.line(img,(0,0),(img.shape[1],img.shape[0]),(0,255,0),3)    #ve duong thang
# cv2.rectangle(img,(0,0),(200,200),(0,255,255),cv2.FILLED)   #ve hinh chu nhat
# cv2.circle(img,(400,50),30,(0,255,0),25)   #ve hinh tron
# cv2.putText(img,"opencv",(250,250),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,255))  #viet chu
# cv2.imshow("ai",img)
# cv2.waitKey(0)

# cat doi tuong
# img = cv2.imread("/home/kiencate/Documents/LearnCodePython/learn opencv/card.jpg")
# (h,w,d)= img.shape
# r= 500/h
# img=cv2.resize(img,(int(w*r),500))
# width,height= 250,500
# pts1= np.float32([[400,66],[650,166],[253,383],[459,489]])
# pts2= np.float32([[0,0],[width,0],[0,height],[width,height]])
# matrix = cv2.getPerspectiveTransform(pts1,pts2)
# imgOutput= cv2.warpPerspective(img,matrix,(width,height))
# cv2.imshow("card", img)
# cv2.imshow("cardOutput", imgOutput)
# cv2.waitKey(0)


#ghep' anh?
# img = cv2.imread(avt)
# (h, w, d) = img.shape
# r = 300.0 / w 
# dim = (int(h * r),300)
# img = cv2.resize(img, dim)
# hor = np.hstack((img,img))
# ver = np.vstack((img,img))
# cv2.imshow("horizontal", hor)
# cv2.imshow("vertical", ver)
# cv2.waitKey(0)


#detect color
# def empty(a):
#     pass

# cv2.namedWindow("TrackBars")
# cv2.resizeWindow("TrackBars",640,240)
# cv2.createTrackbar("Hue Min","TrackBars",0,179,empty)
# cv2.createTrackbar("Hue Max","TrackBars",179,179,empty)
# cv2.createTrackbar("Sat Min","TrackBars",0,255,empty)
# cv2.createTrackbar("Sat Max","TrackBars",255,255,empty)
# cv2.createTrackbar("Val Min","TrackBars",0,255,empty)
# cv2.createTrackbar("Val Max","TrackBars",255,255,empty)

# while True:
#     img=cv2.imread(car)
#     img= cv2.resize(img,(400,320))

#     imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#     h_min = cv2.getTrackbarPos("Hue Min","TrackBars")
#     h_max = cv2.getTrackbarPos("Hue Max","TrackBars")
#     s_min = cv2.getTrackbarPos("Sat Min","TrackBars")
#     s_max = cv2.getTrackbarPos("Sat Max","TrackBars")
#     v_min = cv2.getTrackbarPos("Val Min","TrackBars")
#     v_max = cv2.getTrackbarPos("Val Max","TrackBars")


    
#     lower = np.array([h_min,s_min,v_min])
#     upper = np.array([h_max,s_max,v_max])
#     print(lower)
#     mask = cv2.inRange(imgHSV,lower,upper)
#     imgResult = cv2.bitwise_and(img,img, mask = mask)
#     cv2.imshow("anh goc",img)
    
#     cv2.imshow("mask",mask)
#     cv2.imshow("r",imgResult)
#     k = cv2.waitKey(1) & 0xFF
#     if k == 27:
#         break


#detect shape
# def getContours(img):
#     contours,hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
        
#         if area>10:
#             cv2.drawContours(imgContour,cnt,-1,(255,0,0),3)
#             peri = cv2.arcLength(cnt,True)
            
#             approx = cv2.approxPolyDP(cnt,0.02*peri,True)
        
#             objCor = len(approx)
#             x,y,w,h = cv2.boundingRect(approx)

#             cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)

# img = cv2.imread(shape)
# img = cv2.resize(img, (800,400))
# imgContour = img.copy()

# imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# imgBlur = cv2.GaussianBlur(imgGray,(7,7),1)
# imgCanny = cv2.Canny(imgBlur,50,50)
# getContours(imgCanny)
# cv2.imshow("canny",imgCanny)
# cv2.imshow("contour",imgContour)
# cv2.imshow("gray",imgGray)
# cv2.waitKey(0)

#project1
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,640)

myColors = [24,18,195,77,83,255]
printcolor = [0,0,255]

def findColor(img,myColor):
    imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    
    lower = np.array(myColor[0:3])
    upper = np.array(myColor[3:6])
    mask = cv2.inRange(imgHSV,lower,upper)
    cv2.imshow("img",mask)
    x,y=getContours(mask)
    cv2.circle(imgResult,(x,y),10,(255,0,0),cv2.FILLED)
    cv2.circle(img1,(x,y),10,(255,0,0),cv2.FILLED)
    

def getContours(img):
    contours,hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    x,y,w,h = 0,0,0,0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if area>100:
            #cv2.drawContours(imgResult,cnt,-1,(255,0,0),3)
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)      
            objCor = len(approx)
            x,y,w,h = cv2.boundingRect(approx)
    return x+w//2,y
img1 = np.zeros((512,512,3),np.uint8)

while True:
    success, img = cap.read()
    imgResult = img.copy()
    findColor(img,myColors)
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    cv2.imshow("Videohai",imgResult)
    cv2.imshow("Videohai1",img1)
    if cv2.waitKey(1) & 0xFF == 27:
        break



