import cv2
import proj2
import sys

source_image = "camera_cal/calibration2.jpg"
dest_image = "output_images/calibration2_undistorted.jpg"
if len(sys.argv) >= 2:
    source_image = sys.argv[1]

if len(sys.argv) >= 3:
    dest_image = sys.argv[2]

camera_matrix, distortion_coeffs = \
    proj2.get_camera_data("camera_cal", "calibration*.jpg", (9,6), "output_images", False)

# Test undistortion on an image
img = cv2.imread(source_image)
undistorted = cv2.undistort(img, camera_matrix, distortion_coeffs, None, camera_matrix)
cv2.imwrite(dest_image, undistorted, None)
