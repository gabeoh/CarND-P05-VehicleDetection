#%% Initialization
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

from my_util import print_section_header, determine_color_converter
import p05_04_determine_slide_window as deter_win
import p05_02_feature_extraction as feat_ext


#%% Step 5 - Slide Window Vehicle Search

def search_vehicle(img, y_start, y_stop,
                   classifier, feature_scaler,
                   cspace, spatial_feat, hist_feat, hog_feat,
                   spatial_size, hist_bins,
                   orient, pix_per_cell, cell_per_block, hog_channel):
    # Discard the upper region where vehicles do not appear
    img = img[y_start:y_stop, :, :]
    # Convert color space
    if cspace != 'RGB':
        color_converter = determine_color_converter('RGB', cspace)
        img = cv2.cvtColor(img, color_converter)

    bbox_list = []
    win_size_train = 64
    window_metrics = deter_win.get_window_metrics(y_start=0)
    for win_size, x_start_stop, y_start_stop, overlap in window_metrics:
        # Compute scale
        scale = win_size / win_size_train
        x_span_scale = np.int((x_start_stop[1] - x_start_stop[0]) / scale)
        y_span_scale = np.int((y_start_stop[1] - y_start_stop[0]) / scale)

        # Scale image
        img_search = img[y_start_stop[0]:y_start_stop[1], x_start_stop[0]:x_start_stop[1]]
        if scale == 1:
            img_search = np.copy(img_search)
        else:
            img_h, img_w = img_search.shape[0:2]
            img_search = cv2.resize(img_search, (np.int(img_w / scale), np.int(img_h / scale)))

        # Compute individual channel HOG features for the entire image
        hog_features_overall = []
        for i in range(img_search.shape[2]):
            features = feat_ext.get_hog_features(img_search[:, :, i], orient, pix_per_cell, cell_per_block, vis=False,
                                                 feature_vec=False)
            hog_features_overall.append(features)

        # Define blocks and steps as above
        nb_blocks_per_win = (win_size_train // pix_per_cell) - cell_per_block + 1
        nb_blocks_x = (x_span_scale // pix_per_cell) - cell_per_block + 1
        nb_blocks_y = (y_span_scale // pix_per_cell) - cell_per_block + 1
        # Movement in number of cells per slide window step
        cells_per_step = np.int(win_size_train * (1 - overlap) // pix_per_cell)
        # Number of sliding window steps in X and Y directions (slide through HOG feature blocks)
        nb_steps_x = (nb_blocks_x - nb_blocks_per_win) // cells_per_step + 1
        nb_steps_y = (nb_blocks_y - nb_blocks_per_win) // cells_per_step + 1

        for step_y in range(nb_steps_y):
            for step_x in range(nb_steps_x):
                # Window start position in cells (or, in block indexes (first 2-dimensions) of HOG features)
                pos_y = step_y * cells_per_step
                pos_x = step_x * cells_per_step
                # Window start position in pixels
                x_left = pos_x * pix_per_cell
                y_top = pos_y * pix_per_cell

                # Extract the image patch
                img_win = img_search[y_top:y_top+win_size_train, x_left:x_left+win_size_train]

                features = []
                if spatial_feat == True:
                    # Extract binned spatial features
                    spatial_features = feat_ext.bin_spatial(img_win, size=spatial_size)
                    features.append(spatial_features)
                if hist_feat == True:
                    # Color histograms
                    hist_features = feat_ext.color_hist(img_win, nbins=hist_bins)
                    features.append(hist_features)
                if hog_feat == True:
                    # Extract HOG features for this patch
                    hog_features = []
                    for i in range(len(hog_features_overall)):
                        hog_feats = hog_features_overall[i][
                                       pos_y:pos_y + nb_blocks_per_win,
                                       pos_x:pos_x + nb_blocks_per_win].ravel()
                        hog_features.append(hog_feats)
                    features.append(np.hstack(hog_features))
                test_features = np.hstack(features).reshape(1, -1)
                # Scale features and make a prediction
                test_features = feature_scaler.transform(test_features)
                test_prediction = classifier.predict(test_features)

                if test_prediction == 1:
                    x_left_orig = np.int(x_left * scale + x_start_stop[0])
                    y_top_orig = np.int(y_top * scale + y_start + y_start_stop[0])
                    win_size_orig = np.int(win_size)
                    bbox_list.append((
                        (x_left_orig, y_top_orig),
                        (x_left_orig + win_size_orig, y_top_orig + win_size_orig)))
    return bbox_list


def perform_slide_window_search(img_dir, img_files, out_dir, pickle_file):
    print_section_header("Slide Window Vehicle Search")

    # Load trained classifier
    with open(pickle_file, 'rb') as in_file:
        class_pickle = pickle.load(in_file)

    # Determine list of files to process
    if len(img_files) == 0:
        img_files = sorted(os.listdir(img_dir))
        img_files = [f for f in img_files if not f.startswith('.')]

    for img_file in img_files:
        # Read an image file and correct distortions
        img_name = img_file.split('.')[0]
        img_path = img_dir + img_file
        img = mpimg.imread(img_path)
        bbox_list = search_vehicle(img, y_start=360, y_stop=680,
                       classifier=class_pickle['classifier'], feature_scaler=class_pickle['feature_scaler'],
                       cspace=class_pickle['cspace'],
                       spatial_feat=class_pickle['spatial_feat'],
                       hist_feat=class_pickle['hist_feat'], hog_feat=class_pickle['hog_feat'],
                       spatial_size=class_pickle['spatial_size'], hist_bins=class_pickle['hist_bins'],
                       orient=class_pickle['orient'], pix_per_cell=class_pickle['pix_per_cell'],
                       cell_per_block=class_pickle['cell_per_block'], hog_channel=class_pickle['hog_channel'])

        for bbox in bbox_list:
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)

        out_file = out_dir + img_name + '.jpg'
        print("Store bonding box image to {}".format(out_file))
        cv2.imwrite(out_file, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    # Step 5 - Slide Window Vehicle Search
    undistorted_img_dir = '../output_images/undistorted/'
    slide_search_dir = '../output_images/slide_search/'
    pickle_file = '../results/classifier_YCrCb_sp32_hist32_hog_9_8_2_ALL.p'
    perform_slide_window_search(undistorted_img_dir, [], slide_search_dir, pickle_file)
