import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import numpy as np

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    if orient == 'x':
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    else:
        sobel = None

    abs_sobel = np.absolute(sobel)
    scaled = np.uint8(255*abs_sobel/np.max(abs_sobel))

    # Apply threshold
    grad_binary = np.zeros_like(scaled)
    grad_binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1

    return grad_binary


def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    mag_sobel = np.sqrt(sobelx**2 + sobely**2)
    scaled = np.uint8(255*mag_sobel/np.max(mag_sobel))

    # Apply threshold
    mag_binary = np.zeros_like(scaled)
    mag_binary[(scaled >= mag_thresh[0]) & (scaled <= mag_thresh[1])] = 1

    return mag_binary


def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Calculate gradient direction
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    dir_sobel = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    # Apply threshold
    dir_binary = np.zeros_like(dir_sobel)
    dir_binary[(dir_sobel >= thresh[0]) & (dir_sobel <= thresh[1])] = 1

    return dir_binary


image = mpimg.imread('signs_vehicles_xygrad.png')

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
mpimg.imsave('gray.png', gray, cmap='gray')

# Choose a Sobel kernel size
ksize = 15

# Choose a larger odd number to smooth gradient measurements
# Apply each of the thresholding functions
gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(40, 90))
mpimg.imsave('gradx.png', gradx, cmap='gray')

grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(30, 90))
mpimg.imsave('grady.png', grady, cmap='gray')

mag_binary = mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(40, 90))
mpimg.imsave('magnitude.png', mag_binary, cmap='gray')

dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=(0.7, 1.3))
mpimg.imsave('direction.png', dir_binary, cmap='gray')

combined = np.zeros_like(dir_binary)
combined[((gradx >= 0.8) & (grady <= 0.2)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
mpimg.imsave('combined.png', combined, cmap='gray')

