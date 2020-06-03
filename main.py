import argparse

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

def run(args):
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
        args.recalibrate
    )

    if args.images:
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
            write_image(output_dir, source_image_filename, "lane_lines", imgs["lane_lines"])
            write_image(output_dir, source_image_filename, "layer", imgs["layer"])
            write_image(output_dir, source_image_filename, "final", imgs["final"])

    if args.video:
        video_filename = args.video
        cap = cv2.VideoCapture(video_filename)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        counter = 0
        out = cv2.VideoWriter(os.path.join(output_dir, video_filename),
                              fourcc, fps, (frame_width, frame_height))
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                frame_out = proj2.process_image(frame, camera_matrix, distortion_coeffs)
                out.write(frame_out["final"])
            else:
                break

            counter += 1
            if (counter % fps) == 0:
                print("{} seconds of video processed".format(counter // fps))

        cap.release()
        out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images and/or videos.")
    parser.add_argument(
        "--recalibrate",
        help="Force a rerun of camera calibration.",
        action="store_true"
    )
    parser.add_argument(
        "--images",
        help="Process images found in directory test_images writing outputs to output_images.",
        action="store_true"
    )
    parser.add_argument(
        "--video",
        help="Process the named video, writing output to output_images> Note only one video can be specified."
    )
    args = parser.parse_args()
    run(args)
