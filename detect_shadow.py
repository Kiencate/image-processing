import cv2
import numpy as np
import math
imgTest = "/home/kiencate/image-processing/oiltank.png"
# h_cam : chieu cao camera
# h,w_corner : goc mo cua camera theo huong doc va ngang
# h,w_img : kich co anh chup
h_cam = 100
h_corner = w_corner = 60
h_img = w_img = 800
cres = [[]]


def getContours(img):
    contours,hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    a=0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if area>300:
            cir = []
            peri = cv2.arcLength(cnt,True)  
            cv2.drawContours(imgContour,cnt,-1,(255,0,0),2)         
            approx = cv2.approxPolyDP(cnt,0.03*peri,True)       
            x,y,w,h = cv2.boundingRect(approx)
            # cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)
            # cv2.putText(imgContour,str(c),(int(a),int(b)),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,255))
            
            # xac dinh 2 diem canh
            max = 0
            diem1 = diem2 = []
            trungdiem1 = trungdiem2=[]
            for i in approx:
                for j in approx:
                    dist = np.linalg.norm(i-j)
                    if dist>max:
                        max = dist
                        diem1 = i
                        diem2 = j
            cv2.circle(imgContour,(diem1[0][0],diem1[0][1]),1,(0,255,255),3)
            cv2.circle(imgContour,(diem2[0][0],diem2[0][1]),1,(0,255,255),3)  
            trungdiem = (diem1+diem2)//2
            
            #xac dinh 2 diem tren duoi
            
            min = max
            for i in cnt:
                dist = np.linalg.norm(i-trungdiem)
                if dist < min:
                    trungdiem1 = i
                    min = dist
            cv2.circle(imgContour,(trungdiem1[0][0],trungdiem1[0][1]),1,(0,255,255),3)
            min = np.linalg.norm(diem1 - trungdiem1) + np.linalg.norm(trungdiem1- trungdiem) - np.linalg.norm(diem1- trungdiem)
            for i in cnt:
                if i[0][0] == trungdiem1[0][0] and i[0][1] == trungdiem1[0][1] :
                    continue
                dist0 = np.linalg.norm(trungdiem1- trungdiem)
                dist1 = np.linalg.norm(i- trungdiem)
                dist2 = np.linalg.norm(i - trungdiem1)
                if (dist2 + dist0- dist1)< min :
                    min = dist2 + dist0- dist1
                    trungdiem2 = i
            cv2.circle(imgContour,(trungdiem2[0][0],trungdiem2[0][1]),1,(0,255,255),3)
            print(diem1)
            #them 4 diem vao mang
            cres.extend(diem1)
            cres.extend(diem2)
            cres.extend(trungdiem1)
            cres.extend(trungdiem2)






img=cv2.imread(imgTest)
imgContour = img.copy()
imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lower = np.array([80,13,51])
upper = np.array([134,255,134])

mask = cv2.inRange(imgHSV,lower,upper)
getContours(mask)

# ve vong tron
if cres[3][1]<cres[7][1]:
    tam = (cres[3]+cres[8])//2
    r = int(np.linalg.norm(tam - cres[3]))
else:
    tam = (cres[4]+cres[7])//2
    r = int(np.linalg.norm(tam - cres[4]))
for i in range(9):
    print(cres[i])
cv2.circle(imgContour,(tam[0],tam[1]),r,(0,0,255),2)
 
cv2.imshow("a",imgContour)
cv2.imshow("mask",mask)
k= cv2.waitKey(0)



