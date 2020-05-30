import pickle
import os
import glob
import numpy as np
import cv2

camera_calib_pickle_filename = "camera_calib_pickle.p"
camera_calib_matrix_key = "mtx"
camera_calib_distortion_coeffs_key = "dist"

def get_camera_data(
        calibration_image_dir,
        calibration_image_filename_glob,
        calibration_image_size,
        output_dir,
        recalibrate
):
    camera_calib_pickle_path = os.path.join(output_dir, camera_calib_pickle_filename)

    if os.path.exists(camera_calib_pickle_path) and not recalibrate:
        print("Reading calibration parameters from {}".format(camera_calib_pickle_path))
        dist_pickle = pickle.load(open(camera_calib_pickle_path, "rb"))
        camera_matrix = dist_pickle["mtx"]
        distortion_coeffs = dist_pickle["dist"]

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
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open(camera_calib_pickle_path, "wb"))

    return (mtx, dist)
