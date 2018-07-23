#%% Initialization
import sys
import cv2
import glob
import numpy as np
import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from my_util import print_section_header, determine_color_converter
from p05_02_feature_extraction import extract_features


#%% Step 3 - Train classifier

def extract_training_features(
        vehicle_img_dir, non_vehicle_img_dir, cspace,
        spatial_feat, hist_feat, hog_feat, spatial_size, hist_bins,
        orient, pix_per_cell, cell_per_block, hog_channel):

    vehicle_imgs = glob.glob(vehicle_img_dir + '**/*.png', recursive=True)
    non_vehicle_imgs = glob.glob(non_vehicle_img_dir + '**/*.png', recursive=True)
    print("* Number Of Vehicle Training Images: {}".format(len(vehicle_imgs)))
    print("* Number Of Non-Vehicle Training Images: {}".format(len(non_vehicle_imgs)))

    print("* Color Space: {}".format(cspace))
    print("* Spatial Features: {}\n  - size: {}".format(spatial_feat, spatial_size))
    print("* Color Histogram: {}\n  - Number of bins: {}".format(hist_feat, hist_bins))
    print("* HOG Features: {}".format(hog_feat))
    print("  - using {} orientations, {} pixels per cell, and {} cells per block on '{}' HOG channel(s)".format(
        orient, pix_per_cell, cell_per_block, hog_channel))

    # Extract features from training data
    start_time = time.perf_counter()
    print('\nStart feature extraction on dataset (at {:.3f})'.format(start_time))
    vehicle_features = []
    non_vehicle_features = []
    for features, imgs in zip((vehicle_features, non_vehicle_features), (vehicle_imgs, non_vehicle_imgs)):
        for img_path in imgs:
            # Make sure to read PNG images in [0, 255]
            # cv2.imread reads image in BGR [0, 255]
            color_converter = determine_color_converter('BGR', cspace)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, color_converter)
            file_features = extract_features(img,
                    spatial_feat=spatial_feat, spatial_size=spatial_size,
                    hist_feat=hist_feat, hist_bins=hist_bins,
                    hog_feat=hog_feat, orient=orient, pix_per_cell=pix_per_cell,
                    cell_per_block=cell_per_block, hog_channel=hog_channel)
            features.append(file_features)
    end_time = time.perf_counter()
    print('Completed feature extraction on dataset {:.3f}s (at {:.3f})'.format(end_time - start_time, end_time))
    sys.stdout.flush()

    # Create an array stack of feature vectors
    X = np.vstack((vehicle_features, non_vehicle_features)).astype(np.float64)

    # Define the labels vector
    y = np.hstack((np.ones(len(vehicle_features)), np.zeros(len(non_vehicle_features))))

    return X, y

def train_classifier(X, y):

    # Split up data into randomized training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    assert len(X) == len(y)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    print("* Total Dataset: {}".format(len(X)))
    print("* Total Training Dataset: {}".format(len(X_train)))
    print("* Total Test Dataset: {}".format(len(X_test)))

    # Perform per-feature normalization on train and test dataset
    # Make sure to compute the scaler using train set only, and apply scaling
    # on train and test set separately to avoid peeking test data.
    feature_scaler = StandardScaler().fit(X_train)
    # Apply the scaler to features
    X_train = feature_scaler.transform(X_train)
    X_test = feature_scaler.transform(X_test)

    # Train LinearSVC classifier (SVC: Support Vector Classification)
    start_time = time.perf_counter()
    print("\nStart classifier training (at {:.3f})".format(start_time))
    svc = LinearSVC()
    svc.fit(X_train, y_train)
    end_time = time.perf_counter()
    print("Completed classifier training {:.3f}s (at {:.3f})".format(end_time - start_time, end_time))

    # Check the accuracy score of the SVC
    start_time = time.perf_counter()
    print("\nStart checking accuracy score (at {:.3f})".format(start_time))
    accuracy_score = svc.score(X_test, y_test)
    end_time = time.perf_counter()
    print("Completed checking accuracy score {:.3f}s (at {:.3f})".format(end_time - start_time, end_time))
    print("* Test Accuracy of SVC: {:.1%}".format(accuracy_score))

    # Check the prediction time for a single sample
    start_time = time.perf_counter()
    print("\nStart prediction time estimation (at {:.3f})".format(start_time))
    n_predict = 10
    prediction = svc.predict((X_test[:n_predict]))
    end_time = time.perf_counter()
    print("Completed prediction time estimation {:.3f}s (at {:.3f})".format(end_time - start_time, end_time))
    print("* Labels    : {}".format(y_test[:n_predict]))
    print("* Preditions: {}".format(prediction))
    print("* Elapsed Time ({} predictions): {:.5f}s".format(n_predict, end_time - start_time))
    sys.stdout.flush()

    return svc, feature_scaler, accuracy_score


def perform_classifier_training(
        vehicle_img_dir, non_vehicle_img_dir, results_dir='./', pickle_result=False,
        cspace='YCrCb', spatial_feat=True, hist_feat=True, hog_feat=True,
        spatial_size=(32, 32), hist_bins = 32,
        orient=9, pix_per_cell=8, cell_per_block=2, hog_channel='ALL'):

    print_section_header("Train Classifier")
    X, y = extract_training_features(vehicle_img_dir, non_vehicle_img_dir, cspace,
                                     spatial_feat, hist_feat,hog_feat, spatial_size,
                                     hist_bins, orient, pix_per_cell, cell_per_block, hog_channel)
    classifier, feature_scaler, accuracy_score = train_classifier(X, y)

    trained_classifier = {
        'classifier': classifier,
        'feature_scaler': feature_scaler,
        'accuracy': accuracy_score,
        'cspace': cspace,
        'spatial_feat': spatial_feat,
        'hist_feat': hist_feat,
        'hog_feat': hog_feat,
        'spatial_size': spatial_size,
        'hist_bins': hist_bins,
        'orient': orient,
        'pix_per_cell': pix_per_cell,
        'cell_per_block': cell_per_block,
        'hog_channel': hog_channel
    }
    if pickle_result:
        suffix = '_' + cspace
        suffix += "_sp{}".format(spatial_size[0]) if spatial_feat else ''
        suffix += "_hist{}".format(hist_bins) if hist_feat else ''
        suffix += "_hog_{}_{}_{}_{}".format(orient, pix_per_cell, cell_per_block, hog_channel) if hog_feat else ''

        pickle_file = results_dir + 'classifier' + suffix + '.p'
        print("Pickle the trained classifier to {}".format(pickle_file))
        with open(pickle_file, 'wb') as out_file:
            pickle.dump(trained_classifier, out_file)
    return trained_classifier


if __name__ == '__main__':
    # Step 3 - Train classifier
    train_img_dir = '../training_images/'
    results_dir = '../results/'
    vehicle_img_dir = train_img_dir + 'vehicles/'
    non_vehicle_img_dir = train_img_dir + 'non-vehicles/'

    param_space = (
        ('YCrCb', True, True, True, (32, 32), 32, 9, 8, 2, 'ALL'),
        ('LUV', True, True, True, (32, 32), 32, 9, 8, 2, 'ALL'),
        ('YUV', True, True, True, (32, 32), 32, 9, 8, 2, 'ALL'),
        ('HSV', True, True, True, (32, 32), 32, 9, 8, 2, 'ALL'),
        ('HLS', True, True, True, (32, 32), 32, 9, 8, 2, 'ALL'),
        ('YCrCb', True, True, True, (32, 32), 32, 9, 8, 2, 0),
        ('YCrCb', True, True, True, (32, 32), 32, 9, 8, 2, 1),
        ('YCrCb', True, True, True, (32, 32), 32, 9, 8, 2, 2),
        ('YCrCb', True, False, False, (32, 32), 32, 9, 8, 2, 'ALL'),
        ('YCrCb', False, True, False, (32, 32), 32, 9, 8, 2, 'ALL'),
        ('YCrCb', False, False, True, (32, 32), 32, 9, 8, 2, 'ALL'),
    )
    for params in param_space:
        perform_classifier_training(
            vehicle_img_dir, non_vehicle_img_dir, results_dir, pickle_result=True,
            cspace=params[0], spatial_feat=params[1], hist_feat=params[2], hog_feat=params[3],
            spatial_size=params[4], hist_bins=params[5],
            orient=params[6], pix_per_cell=params[7], cell_per_block=params[8], hog_channel=params[9])

