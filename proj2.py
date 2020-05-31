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


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def process_image(img, camera_matrix, distortion_coeffs):
    img_width, img_height = img.shape[1], img.shape[0]

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
    mag_binary = mag_thresh(s_channel, sobel_kernel=ksize, mag_thresh=(20, 150))
    dir_binary = dir_threshold(s_channel, sobel_kernel=ksize, thresh=(0.7, 1.3))
    s_binary = img_threshold(s_channel, (100, 255))

    combined = np.zeros_like(dir_binary)
    combined[(mag_binary == 1) | (s_binary == 1)] = 1

    # create region of interest for perspective transform
    upper_y = int(0.65 * img_height)
    lower_y = img_height
    img_width_mid = img_width // 2
    upper_x_margin = 80
    lower_x_margin = 470

    src_points = [
        (img_width_mid - lower_x_margin, lower_y),
        (img_width_mid - upper_x_margin, upper_y),
        (img_width_mid + upper_x_margin, upper_y),
        (img_width_mid + lower_x_margin, lower_y),
    ]
    roi_lines = [
        [src_points[0][0], src_points[0][1], src_points[1][0], src_points[1][1]],
        [src_points[1][0], src_points[1][1], src_points[2][0], src_points[2][1]],
        [src_points[2][0], src_points[2][1], src_points[3][0], src_points[3][1]],
        [src_points[3][0], src_points[3][1], src_points[0][0], src_points[0][1]],
    ]

    tmp = np.copy(img)
    draw_lines(tmp, [roi_lines])

    dst_points = [
        [img_width // 4, img_height],
        [img_width // 4, 0],
        [3 * img_width // 4, 0],
        [3 * img_width // 4, img_height],
    ]

    M = cv2.getPerspectiveTransform(np.float32(src_points), np.float32(dst_points))
    perspective = cv2.warpPerspective(combined, M, (img_width, img_height))

    return {
        "undist": undistorted,
        "gray": gray,
        "hls": hls,
        "sbin": s_binary,
        "gradx": gradx,
        "grady": grady,
        "gradm": mag_binary,
        "gradd": dir_binary,
        "gradc": combined,
        "roi": tmp,
        "perspective": perspective,
    }


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    img_height = binary_warped.shape[0]

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(img_height // nwindows)

    # Identify the x and y indices of all nonzero pixels in the image
    # nonzero is a tuple of two arrays, rows and cols, such that
    # binary_warped[rows[i], cols[i]] is non zero.
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0]) # rows
    nonzerox = np.array(nonzero[1]) # cols

    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive the row, col indices in binary_warped
    # corresponding to the left and right lane pixels
    # binary_warped[nonzeroy[left_lane_inds[i]], nonzerox[left_lane_inds[i]]]
    # will be a pixel of the left lane line ... similarly for right_lane_inds
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        # When window = 0, we have a window at the bottom of the image
        # win_y_high = img_height and win_y_low = img_height - window_height
        win_y_low = img_height - (window + 1) * window_height
        win_y_high = img_height - window * window_height

        # calculate the x values of the window corners
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        ### Identify the nonzero pixels in x and y within the window ###
        # This is a multi-stepper:
        # 1) Each conditional (e.g., nonzeroy >= win_y_low) yields a boolean array of True/False
        # 2) The conjunctions (&) yield an array, A, of True/False such that if index i is True
        # then binary_warped[nonzeroy[A[i]], nonzerox[A[i]]] is a nonzero pixel within the window
        # being analysed.
        # 3) A.nonzero()[0] returns an array, B, containing the indices of those locationas that are True,
        # i.e., for all i in 0..len(B), binary_warped[nonzeroy[B[i]], nonzerox[B[i]]] is a
        # nonzero pixel within the window being analysed.). nonzero() collects the indices containing True
        # as True is evaluated to non-zero (False evaluates to zero).
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # nonzerox[good_..._inds] yields indexes of columns that contain non-zero pixels
        # within the window of binary_warped. Taking the mean of these values returns the
        # 'middle' column to use as the centre of the next window.
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    # for all i in 0..len(left_lane_inds), binary_warped[lefty[i], leftx[i]] is a pixel representing
    # the left lane line
    # for all j in 0..len(right_lane_inds), binary_warped[righty[j], rightx[j]] is a pixel representing
    # the right lane line
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # These fit a second order polynomial Ay**2 + By + C where
    # A == left_fit[0], B == left_fit[1], C == left_fit[2] (similar for right_fit)
    # Passing 'y' into polyfit where x would normally go because the lane lines
    # are almost vertical
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    img_height = binary_warped.shape[0]

    # This creates an array of integers from 0 to img_height-1, i.e.,
    # len(ploty) == img_height. Then, the x values for each y is computed using
    # the polynomial parameters computed above
    ploty = np.linspace(0, img_height - 1, img_height)
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    ## Visualization ##
    # Colours in the pixels identified as belonging to the left and right lanes
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    left_poly_points = np.int32(np.column_stack((left_fitx, ploty)))
    right_poly_points = np.int32(np.column_stack((right_fitx, ploty)))
    cv2.polylines(out_img, [left_poly_points, right_poly_points], False, [0,255,255])

    # for i in range(img_height):
    #     cv2.line(out_img, tuple(left_poly_points[i]), tuple(right_poly_points[i]), [0,255,0])

    return left_fit, right_fit, out_img

def calculate_curve_radius(left_fit, right_fit, at_y):
    left_curve_radius = ((1 + (2*left_fit[0]*at_y + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curve_radius = ((1 + (2*right_fit[0]*at_y + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

    return left_curve_radius, right_curve_radius

def fit_poly_real(binary_warped, ym_per_pix):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, _ = find_lane_pixels(binary_warped)

    # first compute in pixel space to get the number of pixels between
    # left and right llnes
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    img_height = binary_warped.shape[0]
    # Generate x and y values for plotting
    img_height = binary_warped.shape[0]

    # This creates an array of integers from 0 to img_height-1, i.e.,
    # len(ploty) == img_height. Then, the x values for each y is computed using
    # the polynomial parameters computed above
    ploty = np.linspace(0, img_height - 1, img_height)
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    # Plots the left and right polynomials on the lane lines
    left_poly_points = np.int32(np.column_stack((left_fitx, ploty)))
    right_poly_points = np.int32(np.column_stack((right_fitx, ploty)))
    left_x_at_bottom = left_poly_points[-1][0]
    right_x_at_bottom = right_poly_points[-1][0]
    lane_width_in_pixels = right_x_at_bottom - left_x_at_bottom

    xm_per_pix = 3.7 / lane_width_in_pixels # assumes standard lane width of 3.7 m (12 ft)

    left_fit_real = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_real = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

    # calculate position of car in lane
    lane_width_mid_point = lane_width_in_pixels // 2
    img_width_mid_point = binary_warped.shape[1] // 2
    car_dist_from_lane_centre = (img_width_mid_point - lane_width_mid_point) * xm_per_pix

    return left_fit_real, right_fit_real, car_dist_from_lane_centre

def calculate_real_curve_radius(img):
    img_height = img.shape[0]
    ym_per_pix = 30. / img_height

    left_fit, right_fit, car_dist_from_lane_centre = fit_poly_real(img, ym_per_pix)
    at_y = img_height * ym_per_pix
    left_curve_radius = ((1 + (2*left_fit[0]*at_y + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curve_radius = ((1 + (2*right_fit[0]*at_y + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

    return left_curve_radius, right_curve_radius, car_dist_from_lane_centre

