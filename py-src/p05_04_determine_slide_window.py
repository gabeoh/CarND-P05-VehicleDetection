#%% Initialization
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from my_util import print_section_header


#%% Step 4 - Determine slide window positions

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    draw_img = np.copy(img)
    # Draw each bounding box on top of the copy of the given image
    for bbox in bboxes:
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
    # return the image with boxes drawn
    return draw_img

def find_slide_windows(img, x_start_stop=[None, None], y_start_stop=[None, None],
                       xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    img_h, img_w = img.shape[0:2]
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img_w
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img_h

    # Compute the span of the region to be searched
    x_span = x_start_stop[1] - x_start_stop[0]
    y_span = y_start_stop[1] - y_start_stop[0]

    # Compute the number of pixels per step in x/y
    nb_pix_per_step_x = np.int(xy_window[0] * (1 - xy_overlap[0]))
    nb_pix_per_step_y = np.int(xy_window[1] * (1 - xy_overlap[1]))
    nb_win_x = np.int((x_span - xy_window[0]) / nb_pix_per_step_x) + 1
    nb_win_y = np.int((y_span - xy_window[1]) / nb_pix_per_step_y) + 1

    # Initialize a list to append window positions to
    window_list = []

    # Find x and y window positions by stepping through
    for i_y in range(nb_win_y):
        for i_x in range(nb_win_x):
            # Calculate each window position
            x_start = x_start_stop[0] + i_x * nb_pix_per_step_x
            y_start = y_start_stop[0] + i_y * nb_pix_per_step_y
            window = ((x_start, y_start), (x_start + xy_window[0], y_start + xy_window[1]))
            # Append window position to list
            window_list.append(window)

    # Return the list of windows
    return window_list

def get_window_metrics(y_start=360):
    # window_metrics = [(xy_window, x_start_stop, y_start_stop, overlap), ...]
    window_metrics = [
        (320, [0, 1280], [y_start, y_start + 320], 0.75),
        (192, [24, 1280], [y_start + 24, y_start + 312], 0.75),
        (128, [0, 1280], [y_start + 32, y_start + 288], 0.75),
        (80, [0, 1280], [y_start + 40, y_start + 240], 0.5),
        (64, [0, 1280], [y_start + 40, y_start + 168], 0.5),
    ]
    return window_metrics

def determine_window_positions(img_dir, img_files, out_dir):
    print_section_header("Determine Slide Window Locations")

    # Determine list of files to process
    if len(img_files) == 0:
        img_files = sorted(os.listdir(img_dir))
        img_files = [f for f in img_files if not f.startswith('.')]

    # window_metrics = [(xy_window, x_start_stop, y_start_stop, overlap), ...]
    window_metrics = get_window_metrics()

    for img_file in img_files:
        # Read an image file and correct distortions
        img_name = img_file.split('.')[0]
        img_path = img_dir + img_file
        img = mpimg.imread(img_path)
        img = np.copy(img)

        fig, sub_plts = plt.subplots(2, 3, figsize=(18, 9))
        i_win = 0
        for win_size, x_start_stop, y_start_stop, overlap in window_metrics:
            xy_window = (win_size, win_size)
            window_list = find_slide_windows(img, x_start_stop=x_start_stop, y_start_stop=y_start_stop,
                                             xy_window=xy_window, xy_overlap=(overlap, overlap))
            img_boxed = draw_boxes(img, window_list, color=(0, 0, 255), thick=3)
            img_boxed = draw_boxes(img_boxed, window_list[2:3], color=(255, 0, 0), thick=6)

            sub_plts[i_win // 3, i_win % 3].set_title("Window Size: {}".format(xy_window), fontsize=20)
            sub_plts[i_win // 3, i_win % 3].imshow(img_boxed)
            i_win += 1
        # Save slide window plot
        sub_plts[-1, -1].axis('off')
        fig.tight_layout()
        fig.subplots_adjust(left=0.05, right=0.98, top=1, bottom=0.)
        out_file = "{}{}.jpg".format(out_dir, img_name)
        print("Store the image with slide-window markings to {}".format(out_file))
        fig.savefig(out_file)
        plt.close()

if __name__ == '__main__':
    # Step 4 - Determine slide window positions
    undistorted_img_dir = '../output_images/undistorted/'
    slide_win_dir = '../output_images/slide_win/'
    determine_window_positions(undistorted_img_dir, [], slide_win_dir)
