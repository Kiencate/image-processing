import cv2
import numpy as np
import matplotlib.pyplot as plt


def Sharpen(img, kernel_size=3):
    kernel = []
    if kernel_size == 3:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.int32)
    if kernel_size == 5:
        kernel = np.array([[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1],
                          [-1, -1, 8, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1]])
    img_result = np.copy(img)
    img_result = img_result.astype('int32')
    a = int(kernel_size/2)
    for i in range(a, img.shape[0]-a):
        for j in range(a, img.shape[1]-a):
            kernel_img = np.array(img[(i-a):(i+a+1), (j-a):(j+a+1)], np.int32)
            img_result[i, j] = np.clip(np.sum(kernel*kernel_img), 0, 255)
    return img_result.astype('uint8')


original = cv2.imread("Lena.png", 0)

dft = cv2.dft(np.float32(original), flags=cv2.DFT_COMPLEX_OUTPUT)
fshift = np.fft.fftshift(dft)
rows, cols = original.shape
crow, ccol = int(rows / 2), int(cols / 2)
# low pass filter
mask_low = np.zeros((rows, cols, 2), np.uint8)
mask_low[crow-50:crow+50, ccol-50:ccol+50] = 1
f_low = fshift * mask_low
ishift = np.fft.ifftshift(f_low)
img_back = cv2.idft(ishift)
img_low_pass_filter = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
img_blur = cv2.GaussianBlur(original, (11, 11), 0)
# high pass filter
mask_high = np.ones((rows, cols, 2), np.uint8)
mask_high[crow-1:crow+1, ccol-1:ccol+1] = 0
f_high = fshift * mask_high
ishift = np.fft.ifftshift(f_high)
img_back = cv2.idft(ishift)
img_high_pass_filter = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
img_sharpen = Sharpen(original)
title = ["original image", "low pass filter",
         "Gaussian blur", "original image", "High pass filter","Sharpen"]
img = [original, img_low_pass_filter, img_blur, original, img_high_pass_filter,img_sharpen]
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.title(title[i])
    plt.axis('off')
    plt.imshow(img[i], cmap='gray')

plt.show()
