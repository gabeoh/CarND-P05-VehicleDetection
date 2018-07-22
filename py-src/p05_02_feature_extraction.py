#%% Initialization
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import time
from skimage.feature import hog

from my_util import print_section_header, determine_color_converter


#%% Step 2 - Feature extraction

# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Resize the feature vector
    features = cv2.resize(img, size).ravel()
    return features

# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the RGB channels separately
    hist_ch0 = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    hist_ch1 = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    hist_ch2 = np.histogram(img[:,:,2], bins=nbins, range=bins_range)

    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((hist_ch0[0], hist_ch1[0], hist_ch2[0]))

    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=True,
                     feature_vec=True):

    hog_returns = hog(img, orientations=orient,
                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                      cells_per_block=(cell_per_block, cell_per_block),
                      block_norm="L2-Hys",
                      visualise=vis,
                      transform_sqrt=True,
                      feature_vector=feature_vec)

    # hog_returns = hog_features, hog_image if vis = True
    # hog_returns = hog_features if vis = False
    return hog_returns

# Define a function to extract features from a list of images
def extract_features(img, spatial_size=(32, 32), hist_bins=32,
                     orient=9, pix_per_cell=8, cell_per_block=2, hog_channel='ALL',
                     spatial_feat=True, hist_feat=True, hog_feat=True):

    # Collect feature vectors from spatial, color-histogram, and HOG feature extractions
    features = []
    if spatial_feat == True:
        # Extract binned spatial features
        spatial_features = bin_spatial(img, size=spatial_size)
        features.append(spatial_features)
    if hist_feat == True:
        # Color histograms
        hist_features = color_hist(img, nbins=hist_bins)
        features.append(hist_features)
    if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(img.shape[2]):
                hog_features.append(get_hog_features(img[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(img[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        features.append(hog_features)
    return np.concatenate(features)

def demonstrate_feature_extraction(vehicle_img_dir, non_vehicle_img_dir, out_dir, cspace='YCrCb',
        spatial_size=(32, 32), hist_bins=32, orient=9, pix_per_cell=8, cell_per_block=2, hog_channel='ALL'):

    print_section_header("Feature Extraction")
    print("* Color Space: {}".format(cspace))
    print("* Spatial Feature Size: {}".format(spatial_size))
    print("* Color Histogram Bins: {}".format(hist_bins))
    print("* HOG Features:")
    print("  - Orientations: {}".format(orient))
    print("  - Pixels Per Cell: {}".format(pix_per_cell))
    print("  - Cells Per Block: {}".format(cell_per_block))
    print("  - HOG channel(s): {}".format(hog_channel))

    vehicle_imgs = glob.glob(vehicle_img_dir + '**/*.png', recursive=True)
    non_vehicle_imgs = glob.glob(non_vehicle_img_dir + '**/*.png', recursive=True)
    n_vehicle = len(vehicle_imgs)
    n_non_vehicle = len(non_vehicle_imgs)
    i_vehicle = np.random.randint(n_vehicle)
    i_non_vehicle = np.random.randint(n_non_vehicle)
    print("* Number Of Vehicle Training Images: {}".format(n_vehicle))
    print("* Number Of Non-Vehicle Training Images: {}".format(n_non_vehicle))
    print("* Vehicle Image Sample: {} ({})".format(vehicle_imgs[i_vehicle], i_vehicle))
    print("* Non-Vehicle Image Sample: {} ({})".format(non_vehicle_imgs[i_non_vehicle], i_non_vehicle))

    # Plot the selected train images
    color_converter = determine_color_converter('BGR', cspace)
    img_vehicle = cv2.imread(vehicle_imgs[i_vehicle])
    img_vehicle = cv2.cvtColor(img_vehicle, cv2.COLOR_BGR2RGB)
    img_non_vehicle = cv2.imread(non_vehicle_imgs[i_non_vehicle])
    img_non_vehicle = cv2.cvtColor(img_non_vehicle, cv2.COLOR_BGR2RGB)
    fig_hog, (sub1, sub2) = plt.subplots(1, 2, figsize=(10, 4))
    sub1.imshow(img_vehicle)
    sub1.set_title('Vehicle', fontsize=20)
    sub2.imshow(img_non_vehicle)
    sub2.set_title('Non-Vehicle', fontsize=20)
    fig_hog.tight_layout()
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.1)
    out_file = "{}sample_image_{}_{}.jpg".format(out_dir, i_vehicle, i_non_vehicle)
    plt.savefig(out_file)
    plt.close()

    # Convert image color space
    color_converter = determine_color_converter('RGB', cspace)
    img_vehicle_conv = cv2.cvtColor(img_vehicle, color_converter)
    img_non_vehicle_conv = cv2.cvtColor(img_non_vehicle, color_converter)

    # Extract features from training data
    start_time = time.perf_counter()
    print('\nStart feature extraction on sample images (at {:.3f})'.format(start_time))
    fig_hog, sub_plts_hog = plt.subplots(3, 4, figsize=(20, 15))
    fig_hist, sub_plts_hist = plt.subplots(2, 1, figsize=(10, 10))
    i_img = 0
    for label, img in zip(('Vehicle', 'Non-Vehicle'), (img_vehicle_conv, img_non_vehicle_conv)):
        # HOG features
        for channel in range(img.shape[2]):
            _, hog_image = get_hog_features(
                img[:,:,channel], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=True)
            sub_plts_hog[channel, 2 * i_img].set_title("{} CH-{}".format(label, channel + 1), fontsize=20)
            sub_plts_hog[channel, 2 * i_img].imshow(img[:,:,channel], cmap='gray')
            sub_plts_hog[channel, 2 * i_img + 1].set_title("{} CH-{} HOG".format(label, channel + 1), fontsize=20)
            sub_plts_hog[channel, 2 * i_img + 1].imshow(hog_image, cmap='gray')

        # Color histogram
        hist_features = color_hist(img, nbins=hist_bins)
        sub_plts_hist[i_img].set_title("{} - Color Histogram".format(label), fontsize=20)
        sub_plts_hist[i_img].bar(range(len(hist_features)), hist_features)
        i_img += 1
    # Save hog images
    fig_hog.tight_layout()
    fig_hog.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    out_file = "{}hog_feature_{}_{}.jpg".format(out_dir, i_vehicle, i_non_vehicle)
    fig_hog.savefig(out_file)
    # Save color histogram
    fig_hist.tight_layout()
    out_file = "{}color_hist_{}_{}.jpg".format(out_dir, i_vehicle, i_non_vehicle)
    fig_hist.savefig(out_file)
    plt.close()
    end_time = time.perf_counter()
    print('Completed feature extraction on sample images {:.3f}s (at {:.3f})'.format(end_time - start_time, end_time))

if __name__ == '__main__':
    # Step 2 - Extract features
    train_img_dir = '../training_images/'
    feat_ext_img_dir = '../output_images/feat_extract/'
    vehicle_img_dir = train_img_dir + 'vehicles/'
    non_vehicle_img_dir = train_img_dir + 'non-vehicles/'
    demonstrate_feature_extraction(vehicle_img_dir, non_vehicle_img_dir, feat_ext_img_dir)
