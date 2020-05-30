import matplotlib.image as mpimg
import cv2
import numpy as np
import proj2
import os

test_images_dir = "test_images"
test_image_filenames = [
    "straight_lines1.jpg",
    "straight_lines2.jpg",
    "test1.jpg",
    "test2.jpg",
    "test3.jpg",
    "test4.jpg",
    "test5.jpg",
    "test6.jpg",
]
output_dir = "output_images"

source_image_filename = test_image_filenames[0]
source_image_path = os.path.join(test_images_dir, source_image_filename)
image = cv2.imread(source_image_path)

def create_image_filename(source_filename, dest_filename_suffix):
    root, ext = os.path.splitext(source_filename)
    return "{}-{}{}".format(root, dest_filename_suffix, ext)

def write_image(output_dir, source_image_filename, dest_filename_suffix, image):
    output_path = os.path.join(output_dir, create_image_filename(source_image_filename, dest_filename_suffix))
    cv2.imwrite(output_path, image)

def write_binary_image(output_dir, source_image_filename, dest_filename_suffix, image):
    write_image(output_dir, source_image_filename, dest_filename_suffix, image * 255)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
write_image(output_dir, source_image_filename, "gray", gray)

# Choose a Sobel kernel size
ksize = 5

# Choose a larger odd number to smooth gradient measurements
# Apply each of the thresholding functions
gradx = proj2.abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(40, 120))
write_binary_image(output_dir, source_image_filename, "gradx", gradx)

grady = proj2.abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(30, 90))
write_binary_image(output_dir, source_image_filename, "grady", grady)

mag_binary = proj2.mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(40, 90))
write_binary_image(output_dir, source_image_filename, "gradmag", mag_binary)

dir_binary = proj2.dir_threshold(gray, sobel_kernel=ksize, thresh=(0.7, 1.3))
write_binary_image(output_dir, source_image_filename, "graddir", dir_binary)

combined = np.zeros_like(dir_binary)
combined[((gradx >= 0.8) & (grady <= 0.2)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
write_binary_image(output_dir, source_image_filename, "combined", combined)
