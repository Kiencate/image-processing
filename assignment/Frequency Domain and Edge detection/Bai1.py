from matplotlib import pyplot as plt
import numpy as np
import cv2

img = cv2.imread('Lena.png', 0)
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
mag, phase = cv2.cartToPolar(dft_shift[:, :, 0], dft_shift[:, :, 1])
spec = np.log(mag) / 30
spec = (255*spec).clip(0, 255).astype(np.uint8)

plt.subplot(1, 3, 1)
plt.title("original image")
plt.imshow(img, cmap='gray')

plt.subplot(1, 3, 2)
plt.title("phase")
plt.imshow(phase, cmap='gray')

plt.subplot(1, 3, 3)
plt.title("spectrum")
plt.imshow(spec, cmap='gray')

plt.show()
