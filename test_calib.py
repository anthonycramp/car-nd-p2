import pickle
import cv2
import os

# Read in the saved objpoints and imgpoints
camera_coeffs_pickle_filename = "output_images/camera_coeffs.p"
if not os.path.exists(camera_coeffs_pickle_filename):
    print("Run python calibrate.py first")
    exit(0)

dist_pickle = pickle.load(open(camera_coeffs_pickle_filename, "rb"))
camera_matrix = dist_pickle["mtx"]
distortion_coeffs = dist_pickle["dist"]

# Test undistortion on an image
img = cv2.imread('camera_cal/calibration2.jpg')

undistorted = cv2.undistort(img, camera_matrix, distortion_coeffs, None, camera_matrix)
cv2.imwrite("output_images/calib_example.jpg", undistorted, None)
