import pickle
import os
import glob
import numpy as np
import cv2

def get_camera_data(
        calibration_image_dir,
        calibration_image_filename_glob,
        calibration_image_size,
        output_dir,
        recalibrate
):
    camera_calib_pickle_filename = "camera_calib_pickle.p"
    camera_calib_matrix_key = "mtx"
    camera_calib_distortion_coeffs_key = "dist"

    camera_calib_pickle_path = os.path.join(output_dir, camera_calib_pickle_filename)

    if os.path.exists(camera_calib_pickle_path) and not recalibrate:
        print("Reading calibration parameters from {}".format(camera_calib_pickle_path))
        dist_pickle = pickle.load(open(camera_calib_pickle_path, "rb"))
        camera_matrix = dist_pickle[camera_calib_matrix_key]
        distortion_coeffs = dist_pickle[camera_calib_distortion_coeffs_key]

        return (camera_matrix, distortion_coeffs)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    (nx, ny) = calibration_image_size
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(os.path.join(calibration_image_dir, calibration_image_filename_glob))

    # Step through the list and search for chessboard corners
    img_size = None
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        if img_size is None:
            img_size = (img.shape[1], img.shape[0])

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            print("Could not find ({},{}) corners in chessboard {}.".format(nx, ny, fname))

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    print("Writing camera calibration parameters to {}.".format(camera_calib_pickle_path))
    dist_pickle = {}
    dist_pickle[camera_calib_matrix_key] = mtx
    dist_pickle[camera_calib_distortion_coeffs_key] = dist
    pickle.dump(dist_pickle, open(camera_calib_pickle_path, "wb"))

    return (mtx, dist)

def img_threshold(img, thresh=(0,255)):
    assert(len(img.shape) == 2)
    binary = np.zeros_like(img)
    binary[(img >= thresh[0]) & (img <= thresh[1])] = 1
    return binary

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    #assert(len(img.shape) == 2)
    # Calculate sobel gradient
    sobel = None
    if orient == 'x':
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobel = np.absolute(sobel)
    scaled = np.uint8(255*abs_sobel/np.max(abs_sobel))

    # Apply threshold
    return img_threshold(scaled, thresh)

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    mag_sobel = np.sqrt(sobelx**2 + sobely**2)
    scaled = np.uint8(255*mag_sobel/np.max(mag_sobel))

    # Apply threshold
    return img_threshold(scaled, mag_thresh)

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Calculate gradient direction
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    dir_sobel = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    # Apply threshold
    return img_threshold(dir_sobel, thresh)

def process_image(img, camera_matrix, distortion_coeffs):
    undistorted = cv2.undistort(img, camera_matrix, distortion_coeffs, None, camera_matrix)

    gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)

    hls = cv2.cvtColor(undistorted, cv2.COLOR_BGR2HLS)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    # Choose a Sobel kernel size
    ksize = 5

    # Choose a larger odd number to smooth gradient measurements
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(s_channel, orient='x', sobel_kernel=ksize, thresh=(40, 120))
    grady = abs_sobel_thresh(s_channel, orient='y', sobel_kernel=ksize, thresh=(30, 90))
    mag_binary = mag_thresh(s_channel, sobel_kernel=ksize, mag_thresh=(40, 90))
    dir_binary = dir_threshold(s_channel, sobel_kernel=ksize, thresh=(0.7, 1.3))

    combined = np.zeros_like(dir_binary)
    combined[((gradx >= 0.8) & (grady <= 0.2)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    return {
        "undist": undistorted,
        "gray": gray,
        "hls": hls,
        "gradx": gradx,
        "grady": grady,
        "gradm": mag_binary,
        "gradd": dir_binary,
        "gradc": combined,
    }