import matplotlib.image as mpimg
import cv2
import numpy as np
import proj2
import os

def create_image_filename(source_filename, dest_filename_suffix):
    root, ext = os.path.splitext(source_filename)
    if dest_filename_suffix == "":
        return source_filename
    else:
        return "{}-{}{}".format(root, dest_filename_suffix, ext)

def write_image(output_dir, source_image_filename, dest_filename_suffix, image):
    output_path = os.path.join(output_dir, create_image_filename(source_image_filename, dest_filename_suffix))
    cv2.imwrite(output_path, image)

def write_binary_image(output_dir, source_image_filename, dest_filename_suffix, image):
    write_image(output_dir, source_image_filename, dest_filename_suffix, image * 255)

def run():
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

    camera_matrix, distortion_coeffs = proj2.get_camera_data(
        "camera_cal",
        "calibration*.jpg",
        (9,6),
        "output_images",
        False
    )

    for source_image_filename in test_image_filenames:
        img = cv2.imread(os.path.join(test_images_dir, source_image_filename))
        imgs = proj2.process_image(img, camera_matrix, distortion_coeffs)

        write_image(output_dir, source_image_filename, "", img)
        write_image(output_dir, source_image_filename, "undist", imgs["undist"])
        write_image(output_dir, source_image_filename, "gray", imgs["gray"])
        write_image(output_dir, source_image_filename, "s", imgs["hls"][:,:,2])
        write_binary_image(output_dir, source_image_filename, "sbin", imgs["sbin"])
        write_binary_image(output_dir, source_image_filename, "gradx", imgs["gradx"])
        write_binary_image(output_dir, source_image_filename, "grady", imgs["grady"])
        write_binary_image(output_dir, source_image_filename, "gradm", imgs["gradm"])
        write_binary_image(output_dir, source_image_filename, "gradd", imgs["gradd"])
        write_binary_image(output_dir, source_image_filename, "gradc", imgs["gradc"])
        write_image(output_dir, source_image_filename, "roi", imgs["roi"])
        write_binary_image(output_dir, source_image_filename, "perspective", imgs["perspective"])

        lane_lines = proj2.fit_polynomial(imgs["perspective"])
        write_image(output_dir, source_image_filename, "out", lane_lines)

if __name__ == "__main__":
    run()

