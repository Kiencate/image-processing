import cv2
import numpy as np
from matplotlib import pyplot as plt
def Median(img, kernel_size = 3):
    img_result = np.copy(img)
    a = int(kernel_size/2) 
    for i in range(a,img.shape[0]-a):
        for j in range(a,img.shape[1]-a):
            kernel = np.sort(img[(i-a):(i+a+1),(j-a):(j+a+1)].reshape(kernel_size**2))
            img_result[i,j] = kernel[int(kernel_size**2/2)]     
    return img_result


def GaussSmoothing(img, kernel_size = 3):
    kernel = []
    if kernel_size == 3:
        kernel = np.array([[1,2,1],[2,4,2],[1,2,1]],np.uint32)
    if kernel_size == 5:
        kernel = np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]],np.uint32)
    img_result = np.copy(img)
    a = int(kernel_size/2) 
    for i in range(a,img.shape[0]-a):
        for j in range(a,img.shape[1]-a):
            kernel_img = img[(i-a):(i+a+1),(j-a):(j+a+1)]
            
            img_result[i,j] = int(np.sum(kernel*kernel_img)/np.sum(kernel))   
            
    return img_result.astype('uint8')


def Sharpen(img, kernel_size = 3):
    kernel = []
    if kernel_size == 3:
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],np.int32)
    if kernel_size == 5:
        kernel = np.array([[-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1],[-1,-1,8,-1,-1],[-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1]])
    img_result = np.copy(img)
    img_result = img_result.astype('int32')
    a = int(kernel_size/2) 
    for i in range(a,img.shape[0]-a):
        for j in range(a,img.shape[1]-a):
            kernel_img = np.array(img[(i-a):(i+a+1),(j-a):(j+a+1)],np.int32)
            img_result[i,j] = np.clip(np.sum(kernel*kernel_img),0,255)         
    return img_result.astype('uint8')



#Median filter compare
img = cv2.imread("Median.jpg",0)
# plt.figure("Median filter")
# plt.subplot(231),plt.imshow(img,cmap='gray'),plt.title("original image")
# plt.subplot(232),plt.imshow(Median(img,3),cmap='gray'),plt.title("kernel size =3")
# plt.subplot(233),plt.imshow(Median(img,5),cmap='gray'),plt.title("kernel size =5")
# plt.subplot(234),plt.imshow(img,cmap='gray'),plt.title("original image")
# plt.subplot(235),plt.imshow(cv2.medianBlur(img,3),cmap='gray'),plt.title("cv2 function 3")
# plt.subplot(236),plt.imshow(cv2.medianBlur(img,5),cmap='gray'),plt.title("cv2 function 5")

# #Gaussian Smoothing filter compare
# img = cv2.imread("Lena.png",0)
# plt.figure("Gaussian Smoothing")
# plt.subplot(231),plt.imshow(img,cmap='gray'),plt.title("original image")
# plt.subplot(232),plt.imshow(GaussSmoothing(img,3),cmap='gray'),plt.title("kernel size =3")
# plt.subplot(233),plt.imshow(GaussSmoothing(img,5),cmap='gray'),plt.title("kernel size =5")
# plt.subplot(234),plt.imshow(img,cmap='gray'),plt.title("original image")
# plt.subplot(235),plt.imshow(cv2.GaussianBlur(img,(3,3),0),cmap='gray'),plt.title("cv2 function 3")
# plt.subplot(236),plt.imshow(cv2.GaussianBlur(img,(5,5),0),cmap='gray'),plt.title("cv2 function 5")

#Sharpen filter compare
img = cv2.imread('/home/kiencate/Pictures/under_water.png',0)
# kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],np.int32)
# plt.figure("Sharpen")
# plt.subplot(231),plt.imshow(img,cmap='gray'),plt.title("original image")
# plt.subplot(232),plt.imshow(Sharpen(img,3),cmap='gray'),plt.title("kernel size =3")
# plt.subplot(233),plt.imshow(Sharpen(img,5),cmap='gray'),plt.title("kernel size =5")
# plt.subplot(234),plt.imshow(img,cmap='gray'),plt.title("original image")
# plt.subplot(235),plt.imshow(cv2.filter2D(img,ddepth=-1,kernel=kernel),cmap='gray'),plt.title("cv2 function 3 ")
# plt.show()
#faster way to compute
#reference: https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
def adjust_image_gamma_lookuptable(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    table = np.array([((i / 255.0) ** gamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)
plt.imshow(low_adjusted[:,:,::-1])