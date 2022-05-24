import cv2
from matplotlib import pyplot as plt
import numpy as np
def cv2_imread(file_path, flag=1):
    # Read Picture Data
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flag)
def highPassFiltering(img,size):#Transfer parameters are Fourier transform spectrogram and filter size
    h, w = img.shape[0:2]#Getting image properties
    h1,w1 = int(h/2), int(w/2)#Find the center point of the Fourier spectrum
    # cv2.circle(img,(256,256),20,(0),-1)
    img[h1-int(size/2):h1+int(size/2), w1-int(size/2):w1+int(size/2)] = 0#Center point plus or minus half of the filter size, forming a filter size that defines the size, then set to 0
    return img
def lowPassFiltering(img,size):#Transfer parameters are Fourier transform spectrogram and filter size
    h, w = img.shape[0:2]#Getting image properties
    print(h,w)
    h1,w1 = int(h/2), int(w/2)#Find the center point of the Fourier spectrum
    img2 = np.ones((h, w), np.uint8)
    # cv2.circle(img2,(w1,h1),size,(1),-1)#Define a blank black image with the same size as the Fourier Transform Transfer
    # img2[h1-int(size/2):h1+int(size/2), w1-int(size/2):w1+int(size/2)] = 1
    #Center point plus or minus half of the filter size, forming a filter size that defines the size, then set to 1, preserving the low frequency part
    img3=img2*img #A low-pass filter is obtained by multiplying the defined low-pass filter with the incoming Fourier spectrogram one-to-one.
    return img3
gray = cv2_imread("/home/kiencate/Documents/Mon_Hoc/Xu_ly_anh/assignment/Fundamental/Lena.png", 1)
gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, (640, 420))

# Fourier transform
img_dft = np.fft.fft2(gray)
dft_shift = np.fft.fftshift(img_dft)  # Move frequency domain from upper left to middle

#High pass filter
dft_shift=lowPassFiltering(dft_shift,180)
res = np.log(np.abs(dft_shift))


# Inverse Fourier Transform
idft_shift = np.fft.ifftshift(dft_shift)  #Move the frequency domain from the middle to the upper left corner
ifimg = np.fft.ifft2(idft_shift)  # Fourier library function call
ifimg = np.abs(ifimg)
cv2.imshow("ifimg",np.int8(ifimg))
cv2.imshow("gray",gray)


# Draw pictures
plt.subplot(131), plt.imshow(gray, 'gray'), plt.title('Original Image')
plt.axis('off')
plt.subplot(132), plt.imshow(res, 'gray'), plt.title('High pass filter')
plt.axis('off')
plt.subplot(133), plt.imshow(np.int8(ifimg), 'gray'), plt.title('Effect after filtering')
plt.axis('off')
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()