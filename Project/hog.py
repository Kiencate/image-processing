from skimage import exposure
from skimage import feature
import cv2
import matplotlib.pyplot as plt
img = cv2.imread("/home/kiencate/Documents/Tu_Hoc/image-processing/assignment/Frequency Domain and Edge detection/Lena.png",0)
(H, hogImage) = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
    cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2",
    visualize=True)

hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
hogImage = hogImage.astype("uint8")
 
plt.imshow(hogImage)
plt.show()