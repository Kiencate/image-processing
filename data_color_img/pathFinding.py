import cv2
import numpy as np
import math

img_big = cv2.imread('ori5.jpg')
img0 = img_big[600:800,400:600]
img1 = img_big[400:600,400:600]
img2 = img_big[200:400,600:800]
img3 = img_big[200:400,800:1000]
img4 = img_big[0:200,1000:1200]
img5 = img_big[0:200,1200:1400]
img6 = img_big[0:200,1400:1600]

a0 = np.array([9,0])
b0 = np.array([0,3])
a1 = np.array([9,3])
b1 = np.array([0,9])
a2 = np.array([9,0])
b2 = np.array([0,9])
a3 = np.array([0,0])
b3 = np.array([0,9])
a4 = np.array([9,0])
b4 = np.array([5,9])
a5 = np.array([5,0])
b5 = np.array([5,9])
a6 = np.array([5,0])
b6 = np.array([0,9])

def path_drawing200(img,a,b):
    lower = np.array([33,0,0])
    upper = np.array([255,255,255])
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask =  cv2.inRange(imgHSV,lower,upper)
    matrix = np.zeros((10,10), np.uint8)
    for i in range(10):
        for j in range(10):
            x= np.array(mask[i*20:(i+1)*20, j*20:(j+1)*20])
            if np.mean(x)==255:
                matrix[i][j] = 1
            else:
                matrix[i][j] = 0
    # toa do a:src, b:dst
    
    # matrix1 = np.zeros((abs(a[0]-b[0]+1),abs(a[1]-b[1]+1)), np.uint8)
    if a[0]<b[0]:
        n=1             #di ngang
        if a[1]<b[1]:
            m=1         #trai sang phai
        elif a[1]>b[1]:
            m=0            #phai sang trai
        else:
            m=2            #di doc
    elif a[0]>b[0]:
        n=0     #di len
        if a[1]<b[1]:
            m=1         # trai sang phai 
        elif a[1]>b[1]:
            m=0     # phai sang trai
        else:
            m=2             #di doc
    else:
        n=2             #di ngang ngang
        if a[1]<b[1]:
            m=1         # trai sang phai 
        elif a[1]>b[1]:
            m=0         #phai sang trai  
    path = np.zeros((10,10), np.uint8)
    path_ori = np.zeros((50,2),np.uint8)
    # if a[1]== 9 :
    #     path_ori[0] = [a[1]*20+20, a[0]*20]
    # elif a[1]== 0:
    #     path_ori[0] = [a[1]*20, a[0]*20]
    # elif a[0] ==9:
    #     path_ori[0] = [a[1]*20, a[0]*20+20]
    # elif a[0]==0:
    #     path_ori[0] = [a[1]*20, a[0]*20]
    path_ori[0] = [a[1]*20+10, a[0]*20+10]
    index =1

    if n==1 and m==1: # di ngang, trai sang phai
        i=a[0]
        j=a[1]
        path[i,j] =1
        while path[b[0],b[1]]==0:
            if j == b[1]:
                path[i+1][j]=1
                i+=1 
                path_ori[index] = [j*20+10,i*20+10]
                index +=1
            elif matrix[i][j+1]==1:
                path[i][j+1]=1
                j+=1 
                path_ori[index] = [j*20+10,i*20+10]
                index +=1
            else:
                path[i+1][j]=1
                i+=1
                path_ori[index] = [j*20+10,i*20+10]
                index +=1
    if n==0 and m==1: #di len, trai sang phai
        i=a[0]
        j=a[1]
        path[i,j] =1
        while path[b[0],b[1]] == 0:
            if i==b[0]:
                path[i][j+1]=1
                j+=1
                path_ori[index] = [j*20+10,i*20+10]
                index +=1
            elif matrix[i-1][j]==1:
                path[i-1][j]=1
                i-=1
                path_ori[index] = [j*20+10,i*20+10]
                index +=1
            else:
                path[i][j+1]=1
                j+=1
                path_ori[index] = [j*20+10,i*20+10]
                index +=1
    if n==1 and m==0:   #di ngang, phai sang trai
        i=a[0]
        j=a[1]
        path[i,j] =1
        while path[b[0],b[1]] == 0:
            if j==b[1]:
                path[i+1][j]=1
                i+=1
                path_ori[index] = [j*20+10,i*20+10]
                index +=1
            elif matrix[i][j-1]==1:
                path[i][j-1]=1
                j-=1
                path_ori[index] = [j*20+10,i*20+10]
                index +=1
            else:
                path[i+1][j]=1
                i+=1
                path_ori[index] = [j*20+10,i*20+10]
                index +=1
    if n==1 and m==0:   #di len, phai sang trai
        i=a[0]
        j=a[1]
        path[i,j] =1
        while path[b[0],b[1]] == 0:
            if i==b[0]:
                path[i][-1j]=1
                j-=1
                path_ori[index] = [j*20+10,i*20+10]
                index +=1
            elif matrix[i-1][j]==1:
                path[i-1][j]=1
                i-=1
                path_ori[index] = [j*20+10,i*20+10]
                index +=1
            else:
                path[i][j-1]=1
                j-=1
                path_ori[index] = [j*20+10,i*20+10]
                index +=1
    if n==2:
        if m==0:   # ngang ngang phai sang trai
            i=a[0]
            j=a[1]
            path[i,j] =1 
            while path[b[0],b[1]] == 0:
                j-=1
                path[i,j]=1
                path_ori[index] = [j*20+10,i*20+10]
                index +=1
                
        if m==1: #ngang ngang trai sang phai
            i=a[0]
            j=a[1]
            path[i,j] =1 
            while path[b[0],b[1]] == 0:
                j+=1
                path[i,j]=1
                path_ori[index] = [j*20+10,i*20+10]
                index +=1
    if m==2:
        if n==1: #di thang xuong
            i=a[0]
            j=a[1]
            path[i,j] =1 
            while path[b[0],b[1]] == 0:
                i+=1
                path[i,j]=1
                path_ori[index] = [j*20+10,i*20+10]
                index +=1
        if n==0: #di thang len
            i=a[0]
            j=a[1]
            path[i,j] =1 
            while path[b[0],b[1]] == 0:
                i-=1
                path[i,j]=1
                path_ori[index] = [j*20+10,i*20+10]
                index +=1
  
    cv2.circle(img, (a[1]*20+10,a[0]*20+10), 5,(0,0,255),15) 
    cv2.circle(img, (b[1]*20+10,b[0]*20+10), 5,(0,0,255),15)       
    for i in range(index-1):
        cv2.line(img,path_ori[i],path_ori[i+1],(0,255,0),25)
    
path_drawing200(img0,a0,b0)
path_drawing200(img1,a1,b1)
path_drawing200(img2,a2,b2)
path_drawing200(img3,a3,b3)
path_drawing200(img4,a4,b4)
path_drawing200(img5,a5,b5)
path_drawing200(img6,a6,b6)


cv2.imshow('ori',img_big)
cv2.waitKey(0)