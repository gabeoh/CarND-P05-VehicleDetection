#%% Initialization
import cv2
import numpy as np
from skimage.feature import hog


#%% Step 2a - Feature extraction

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
