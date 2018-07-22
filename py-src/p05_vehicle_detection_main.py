#%% Initialization
import os
import pickle
import argparse
from moviepy.editor import VideoFileClip

from my_util import print_section_header, analyze_test_image
import p05_01_correct_distortion as dist_correct
import p05_02_feature_extraction as feat_ext
import p05_03_train_classifier as train_class

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
binary_lane_dir = output_dir + 'binary_lanes/'
perspective_trans_dir = output_dir + 'perspective/'
slide_win_dir = output_dir + 'slide_win/'
overlay_dir = output_dir + 'overlay/'
video_dst_dir = output_dir + 'video/'


#%% Run lane detection on test images
def detect_vehicle_images(img_files, steps):
    print("\n** Running lane detection on test images **")

    # Determine list of files to process
    if len(img_files) == 0:
        img_files = sorted(os.listdir(test_img_dir))
        img_files = [f for f in img_files if not f.startswith('.')]

    # Step 0 - Analyze test image
    if (not steps) or (0 in steps):
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


#%% Run lane detection on provided video
def detect_vehicle_video(video_files, steps):

    print("\n** Running lane detection on video files **")
    print('Video Files: ', video_files)

    # Load distortion correction parameters
    pickle_file = results_dir + 'camera_cal.p'
    with open(pickle_file, 'rb') as inf:
        camera_cal = pickle.load(inf)
    camera_mtx = camera_cal['camera_matrix']
    dist_coeffs = camera_cal['distortion_coefficients']

    # Get matrices for perspective transform and its reverse operation
    # mtx_trans, mtx_trans_inv = p_trans.compute_perspective_transform_matrix()


    videos = sorted(os.listdir(test_video_dir))
    videos = [f for f in videos if f.endswith('.mp4') and (len(video_files) == 0 or f in video_files)]
    for video_file in videos:
        video_path = test_video_dir + video_file
        video_out_path = video_dst_dir + video_file
        print_section_header("Run lane detection on video - {}".format(video_file), 60)

        # Use subclip() to test with shorter video (the first 5 seconds for example)
        # clip = VideoFileClip(video_path).subclip(38,42)
        clip = VideoFileClip(video_path)
        prev_polys = [None, None]
        # clip_processed = clip.fl_image(lambda img: over_annot.overlay_lane_lines(img, camera_mtx, dist_coeffs, mtx_trans, mtx_trans_inv, prev_polys))

        # Write line detected videos to files
        # clip_processed.write_videofile(video_out_path, audio=False)


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
    print("Running lane detection with arguments: {}".format(args))

    files = [] if args.files is None else args.files
    if args.run_image:
        detect_vehicle_images(files, args.steps)
    if args.run_video:
        detect_vehicle_video(files, args.steps)
