#%% Initialization
import os
import argparse
import cv2
import pickle
from moviepy.editor import VideoFileClip

from my_util import print_section_header, analyze_test_image, draw_bounding_boxes
import p05_01_correct_distortion as dist_correct
import p05_02_feature_extraction as feat_ext
import p05_03_train_classifier as train_class
import p05_04_determine_slide_window as deter_win
import p05_05_slide_window_search as slide_search

project_root_dir = '../'
output_dir = project_root_dir + 'output_images/'
results_dir = project_root_dir + 'results/'
train_img_dir = project_root_dir + 'training_images/'
vehicle_img_dir = train_img_dir + 'vehicles/'
non_vehicle_img_dir = train_img_dir + 'non-vehicles/'
test_img_dir = project_root_dir + 'test_images/'
test_video_dir = project_root_dir + 'test_videos/'
feat_ext_img_dir = output_dir + 'feat_extract/'
undistorted_img_dir = output_dir + 'undistorted/'
slide_win_dir = output_dir + 'slide_win/'
slide_search_dir = output_dir + 'slide_search/'
heat_map_dir = output_dir + 'heat_map/'
detection_dir = output_dir + 'vehicle_detection/'
video_dst_dir = output_dir + 'video/'


#%% Run lane detection on test images
def detect_vehicle_images(img_files, steps):
    print("\n** Running vehicle detection on test images **")

    # Step 0 - Analyze test image
    if (not steps) or (0 in steps):
        # Determine list of files to process
        if len(img_files) == 0:
            img_files = sorted(os.listdir(test_img_dir))
            img_files = [f for f in img_files if not f.startswith('.')]
        img_file = img_files[0]
        print_section_header("Analyze Test Images")
        img_path = test_img_dir + img_file
        analyze_test_image(img_path=img_path)

    # Step 1 - Correct image distortion
    if (not steps) or (1 in steps):
        pickle_file = results_dir + 'camera_cal.p'
        dist_correct.perform_distortion_correction(pickle_file, test_img_dir, img_files, undistorted_img_dir)

    # Step 2 - Extract features
    if (not steps) or (2 in steps):
        feat_ext.demonstrate_feature_extraction(vehicle_img_dir, non_vehicle_img_dir, feat_ext_img_dir)

    # Step 3 - Train classifier
    if (not steps) or (3 in steps):
        train_class.perform_classifier_training(vehicle_img_dir, non_vehicle_img_dir, results_dir)

    # Step 4 - Determine slide window positions
    if (not steps) or (4 in steps):
        deter_win.determine_window_positions(undistorted_img_dir, img_files, slide_win_dir)

    # Step 5 - Slide Window Vehicle Search
    if (not steps) or (5 in steps):
        pickle_file = results_dir + 'classifier_YCrCb_sp32_hist32_hog_9_8_2_ALL.p'
        slide_search.perform_slide_window_search(undistorted_img_dir, img_files, pickle_file, slide_search_dir,
                                                 heat_map_dir, detection_dir)


#%% Run all pipeline steps
def run_all_steps(img, steps, camera_mtx, dist_coeffs, class_pickle, prev_frame_data):

    # Step 1 - Correct image distortion
    if (not steps) or (1 in steps):
        img = dist_correct.correct_image_distortion(img, camera_mtx, dist_coeffs)

    # Step 5 - Slide Window Vehicle Search
    if (not steps) or (5 in steps):
        bbox_list = slide_search.find_vehicle_bounding_boxes(img, class_pickle, prev_frame_data)
        img = draw_bounding_boxes(img, bbox_list)

    return img


#%% Run lane detection on provided video
def detect_vehicle_video(video_files, steps):

    print("\n** Running lane detection on video files **")
    print('Video Files: ', video_files)

    # Load distortion correction parameters
    pickle_file_camera = results_dir + 'camera_cal.p'
    with open(pickle_file_camera, 'rb') as in_file:
        camera_pickle = pickle.load(in_file)
    camera_mtx = camera_pickle['camera_matrix']
    dist_coeffs = camera_pickle['distortion_coefficients']

    # Load trained classifier
    pickle_file_class = '../results/classifier_YCrCb_sp32_hist32_hog_9_8_2_ALL.p'
    with open(pickle_file_class, 'rb') as in_file:
        class_pickle = pickle.load(in_file)

    # Determine list of files to process
    if len(video_files) == 0:
        video_files = sorted(os.listdir(test_video_dir))
        video_files = [f for f in video_files if not f.startswith('.')]

    for video_file in video_files:
        video_path = test_video_dir + video_file
        video_out_path = video_dst_dir + video_file
        print_section_header("Run lane detection on video - {}".format(video_file), 60)

        # Data structure to track information from previous frame
        prev_frame_data = {
            'heat_map': None
        }
        # Use subclip() to test with shorter video (the first 5 seconds for example)
        # clip = VideoFileClip(video_path).subclip(34, 43)
        clip = VideoFileClip(video_path)
        clip_processed = clip.fl_image(
            lambda img: run_all_steps(img, steps, camera_mtx, dist_coeffs, class_pickle, prev_frame_data))

        # Write line detected videos to files
        clip_processed.write_videofile(video_out_path, audio=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect Lane Lines')
    parser.add_argument('-i', '--image', dest='run_image', action='store_true',
                        help='Execute lane detection on test images')
    parser.add_argument('-v', '--video', dest='run_video', action='store_true',
                        help='Execute lane detection on test video')
    parser.add_argument('-f', '--files', dest='files', type=str, nargs='*',
                        help='Provide image/video file(s) to process. Run on all test files when omitted.')
    parser.add_argument('-s', '--steps', dest='steps', type=int, nargs='*',
                        help='Provide steps to execute. Run all steps when omitted.')
    args  = parser.parse_args()
    print("Running vehicle detection with arguments: {}".format(args))

    files = [] if args.files is None else args.files
    if args.run_image:
        detect_vehicle_images(files, args.steps)
    if args.run_video:
        detect_vehicle_video(files, args.steps)
