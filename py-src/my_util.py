import cv2
import numpy as np
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

def print_section_header(title, len_banner=35):
    """
    Helper function to print section header with given title
    :param title:
    :return:
    """
    print()
    print('#' * len_banner)
    print('#', title)
    print('#' * len_banner)

# Analyze image details
def analyze_test_image(img_path):
    img = mpimg.imread(img_path)
    img_y_size, img_x_size = img.shape[0:2]
    print("Image File: {}".format(img_path))
    print("Image Size: {}x{}".format(img_x_size, img_y_size))
    print("Image Min/Max Values: ({}, {})".format(img.min(), img.max()))

def determine_color_converter(cspace_src, cspace_dst):
    converter = None
    if cspace_src == 'RGB':
        if cspace_dst == 'YCrCb':
            converter = cv2.COLOR_RGB2YCrCb
        elif cspace_dst == 'LUV':
            converter = cv2.COLOR_RGB2LUV
        elif cspace_dst == 'YUV':
            converter = cv2.COLOR_RGB2YUV
        elif cspace_dst == 'HLS':
            converter = cv2.COLOR_RGB2HLS
        elif cspace_dst == 'HSV':
            converter = cv2.COLOR_RGB2HSV
    elif cspace_src == 'BGR':
        if cspace_dst == 'YCrCb':
            converter = cv2.COLOR_BGR2YCrCb
        elif cspace_dst == 'LUV':
            converter = cv2.COLOR_BGR2LUV
        elif cspace_dst == 'YUV':
            converter = cv2.COLOR_BGR2YUV
        elif cspace_dst == 'HLS':
            converter = cv2.COLOR_BGR2HLS
        elif cspace_dst == 'HSV':
            converter = cv2.COLOR_BGR2HSV
    return converter

def draw_bounding_boxes(img, bbox_list, color=(0, 0, 255), thickness=6):
    out_img = np.copy(img)
    for bbox in bbox_list:
        cv2.rectangle(out_img, bbox[0], bbox[1], color, thickness)
    return out_img

def compute_curvature_poly2(A, B, y_eval):
    return ((1 + (2 * A * y_eval + B)**2)**1.5) / np.abs(2*A)

def capture_frame(video_path, t, out_file):
    clip = VideoFileClip(video_path)
    clip.save_frame(out_file, t)

# capture_frame('../test_videos/project_video.mp4', 39.6, '../test_images/additional1.jpg')
# capture_frame('../test_videos/project_video.mp4', 41.5, '../test_images/additional2.jpg')
# capture_frame('../test_videos/project_video.mp4', 41.6, '../test_images/additional3.jpg')
