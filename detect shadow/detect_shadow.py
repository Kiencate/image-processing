import numpy as np
import cv2
import math

img = cv2.imread('oitank6.png')
gray = cv2.imread('oitank6.png', 0)
# img = img[0:215, 437:img.shape[1]-1]
img = img[0:215, 207:408]
blur = cv2.bilateralFilter(img,9,75,75)

#############################    HSI CONVERSION    ###########################

blur = np.divide(blur, 255.0)
hsi = np.zeros((blur.shape[0],blur.shape[1],blur.shape[2]),dtype=np.float)
ratio_map = np.zeros((blur.shape[0],blur.shape[1]),dtype=np.uint8)
a= np.zeros((blur.shape[0],blur.shape[1]),dtype=np.float)

for i in range(blur.shape[0]):
    for j in range(blur.shape[1]):
        hsi[i][j][2] = (blur[i][j][0]+blur[i][j][1]+blur[i][j][2])/3
        hsi[i][j][0] = math.acos(((blur[i][j][2]-blur[i][j][1])*(blur[i][j][2]-blur[i][j][0]))/(2*math.sqrt((blur[i][j][2]-blur[i][j][1])*(blur[i][j][2]-blur[i][j][1])+(blur[i][j][2]-blur[i][j][0])*(blur[i][j][1]-blur[i][j][0]))))
        hsi[i][j][1] = 1 - 3*min(blur[i][j][0],blur[i][j][1],blur[i][j][2])/hsi[i][j][2]
        a[i][j] = hsi[i][j][0]/(hsi[i][j][2]) 
        if np.isnan(a[i][j]):
              a[i][j] = np.nan_to_num(a[i][j])
        ratio_map[i][j]=a[i][j]                   

###############################################################################
 
#########################    OTSU'S METHOD    #################################

hist = np.histogram(ratio_map.ravel(),256,[0,256])
ret,th = cv2.threshold(ratio_map,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
mask= cv2.medianBlur(th,9)
###############################################################################



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
            
            #them 4 diem vao mang
            cres.extend(diem1)
            cres.extend(diem2)
            cres.extend(trungdiem1)
            cres.extend(trungdiem2)






imgContour = img.copy()
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
while True:
    
    cv2.imshow("a",imgContour)
    cv2.imshow("mask",img)
    cv2.imshow("mask",mask)
    k= cv2.waitKey(0)
    if k==27:
        break
