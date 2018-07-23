#%% Initialization
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from scipy.ndimage.measurements import label

from my_util import print_section_header, determine_color_converter, draw_bounding_boxes
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
                    bbox_list.append((
                        (x_left_orig, y_top_orig),
                        (x_left_orig + win_size, y_top_orig + win_size)))
    return bbox_list

def add_heat(heat_map, bbox_list, heat_contribution=1, heat_map_factor=1.0):
    # Increment heat value for the areas inside of bounding boxes
    for box in bbox_list:
        heat_map[box[0][1]:box[1][1], box[0][0]:box[1][0]] += heat_contribution * heat_map_factor
    # Return updated heatmap
    return heat_map

def get_bounding_boxes_from_labels(labels):
    bbox_list = []
    for label in range(1, labels[1] + 1):
        # Find pixels with each label value
        label_indexes = (labels[0] == label).nonzero()
        label_coords_x = np.array(label_indexes[1])
        label_coords_y = np.array(label_indexes[0])
        bbox = ((np.min(label_coords_x), np.min(label_coords_y)), (np.max(label_coords_x), np.max(label_coords_y)))
        bbox_list.append(bbox)
    # Return bonding boxes
    return bbox_list

def find_vehicle_bounding_boxes(img, class_pickle, prev_frame_data=None, img_name=None,
                                out_dir_slide=None, out_dir_heat=None, out_dir_detect=None):

    bbox_list = search_vehicle(img, y_start=400, y_stop=656,
                               classifier=class_pickle['classifier'], feature_scaler=class_pickle['feature_scaler'],
                               cspace=class_pickle['cspace'],
                               spatial_feat=class_pickle['spatial_feat'],
                               hist_feat=class_pickle['hist_feat'], hog_feat=class_pickle['hog_feat'],
                               spatial_size=class_pickle['spatial_size'], hist_bins=class_pickle['hist_bins'],
                               orient=class_pickle['orient'], pix_per_cell=class_pickle['pix_per_cell'],
                               cell_per_block=class_pickle['cell_per_block'], hog_channel=class_pickle['hog_channel'])

    # Store raw slide window search results
    if out_dir_slide is not None:
        out_img = draw_bounding_boxes(img, bbox_list)
        out_file = out_dir_slide + img_name + '.jpg'
        print("Store raw sliding search image to {}".format(out_file))
        cv2.imwrite(out_file, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))

    # Check if heat-map is given from previous frame
    heat_contribution_factor = 1.0
    heat_contribution_factor_prev = 0.5
    if prev_frame_data is not None and prev_frame_data['heat_map'] is not None:
        heat_map = prev_frame_data['heat_map'] * heat_contribution_factor_prev
        heat_contribution_factor -= heat_contribution_factor_prev
    else:
        # Initialize heat_map to the same size as img (with zeros)
        heat_map = np.zeros_like(img[:, :, 0]).astype(np.float)

    # Add heat to each box in box list
    heat_multiplier = 12
    heat_map = add_heat(heat_map, bbox_list, heat_multiplier, heat_contribution_factor)
    # Apply threshold to filter out false positives
    heat_threshold = 4 * heat_multiplier
    heat_map[heat_map <= heat_threshold] = 0
    # Store heat-map image
    if out_dir_heat is not None:
        # Normalize heat map to [0, 255] for visualization
        max_heat = np.max(heat_map)
        heat_map_out = heat_map
        if max_heat > 0:
            heat_map_out = (heat_map_out * 255 / max_heat).astype(np.uint8)
        out_file = out_dir_heat + img_name + '.jpg'
        print("Store heat-map image to {}".format(out_file))
        cv2.imwrite(out_file, heat_map_out)
    if prev_frame_data is not None:
        prev_frame_data['heat_map'] = heat_map

    # Find final boxes from heatmap using label function
    box_labels = label(heat_map)
    bbox_list_final = get_bounding_boxes_from_labels(box_labels)
    if out_dir_detect is not None:
        out_img = draw_bounding_boxes(img, bbox_list_final)
        out_file = out_dir_detect + img_name + '.jpg'
        print("Store the image with vehicle detection to {}".format(out_file))
        cv2.imwrite(out_file, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    return bbox_list_final

def perform_slide_window_search(img_dir, img_files, pickle_file, out_dir_slide=None,
                                out_dir_heat=None, out_dir_detect=None):
    print_section_header("Slide Window Vehicle Search")

    # Load trained classifier
    with open(pickle_file, 'rb') as in_file:
        class_pickle = pickle.load(in_file)

    # Determine list of files to process
    if len(img_files) == 0:
        img_files = sorted(os.listdir(img_dir))
        img_files = [f for f in img_files if not f.startswith('.')]

    for img_file in img_files:
        img_name = img_file.split('.')[0]
        img_path = img_dir + img_file
        img = mpimg.imread(img_path)
        find_vehicle_bounding_boxes(img, class_pickle, None, img_name, out_dir_slide, out_dir_heat, out_dir_detect)

if __name__ == '__main__':
    # Step 5 - Slide Window Vehicle Search
    undistorted_img_dir = '../output_images/undistorted/'
    slide_search_dir = '../output_images/slide_search/'
    heat_map_dir = '../output_images/heat_map/'
    detection_dir = '../output_images/vehicle_detection/'
    pickle_file = '../results/classifier_YCrCb_sp32_hist32_hog_9_8_2_ALL.p'
    perform_slide_window_search(undistorted_img_dir, [], pickle_file, slide_search_dir, heat_map_dir, detection_dir)
