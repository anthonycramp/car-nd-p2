import matplotlib.image as mpimg
import cv2
import numpy as np
import proj2

image = mpimg.imread('signs_vehicles_xygrad.png')

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
mpimg.imsave('gray.png', gray, cmap='gray')

# Choose a Sobel kernel size
ksize = 15

# Choose a larger odd number to smooth gradient measurements
# Apply each of the thresholding functions
gradx = proj2.abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(40, 90))
mpimg.imsave('gradx.png', gradx, cmap='gray')

grady = proj2.abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(30, 90))
mpimg.imsave('grady.png', grady, cmap='gray')

mag_binary = proj2.mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(40, 90))
mpimg.imsave('magnitude.png', mag_binary, cmap='gray')

dir_binary = proj2.dir_threshold(gray, sobel_kernel=ksize, thresh=(0.7, 1.3))
mpimg.imsave('direction.png', dir_binary, cmap='gray')

combined = np.zeros_like(dir_binary)
combined[((gradx >= 0.8) & (grady <= 0.2)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
mpimg.imsave('combined.png', combined, cmap='gray')

