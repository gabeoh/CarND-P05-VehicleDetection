# Vechicle Detection And Tracking

## Objective

The goal of this project is to detect and track vehicles in a video.
The following steps are required in order to successfully meet the project
goal.

1. Train a Linear SVM classifier by extracting a Histogram of Oriented
Gradients (HOG), binned color features, and histograms of color from
training image set.
1. Detect vehicles in images using the trained classifier and a sliding-window
technique.
1. Perform the vehicle detection on video stream.
1. Find bounding boxes for detected vehicles, reject outliers and track
the vehicles by creating a heat map of recurring detections across frames.


[//]: # (Image References)

[img_feat_ext_01]: ./output_images/feat_extract/sample_image_3605_981.jpg
[img_feat_ext_02]: ./output_images/feat_extract/color_hist_3605_981.jpg
[img_feat_ext_03]: ./output_images/feat_extract/hog_feature_3605_981.jpg
[img_slide_win_01]: ./output_images/slide_win/test6.jpg
[img_slide_search_01]: ./output_images/slide_search/test6.jpg
[img_heat_map_01]: ./output_images/heat_map/test6.jpg
[img_vehicle_detection_01]: ./output_images/vehicle_detection/test6.jpg

---

## Feature Extraction
The source code for feature extraction is in
[py-src/p05_02_feature_extraction.py](py-src/p05_02_feature_extraction.py).

From provided vehicle and non-vehicle training images,
following features are extracted:
* Histogram of Oriented Gradients (HOG)
* Binned spatial color features
* Histograms of color 

#### Sample Images
The image below illustrate examples of vehicle and non-vehicle training images.
![Sample Images][img_feat_ext_01]


#### Color Histograms
The graph below shows color histograms of above sample images in
`YCrCb` color space, each channel aggregated to 32 bins.
![Color Histogram][img_feat_ext_02]

#### HOG Feature Extraction
Various color spaces and HOG parameters are explored for the feature
extraction.  Finally, the color space of `YCrCb`, and the HOG parameters of
`origentations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`
are selected.
The image below demonstrates HOG feature extractions of the sample images.
![HOG Features][img_feat_ext_03]

---

## Classifier Training
The source code for feature extraction is in
[py-src/p05_03_train_classifier.py](py-src/p05_03_train_classifier.py).

Linear Support Vector Machine (SVM) is selected as a classifier in this
implementation.  The classifier is trained using all three features
(spatial, color histogram, and HOG) extracted from provided vehicle and
non-vehicle training images.

The training images are divided into training (80%) and test (20%) sets.
The classifier is trained on the training set, and then validated on
the test set.

The classifier is trained and tested across various color spaces and
feature parameters in order to determine optimal settings.
The best accuracy is achieved when all three feature extractions
are utilized, and all three color channels are used for HOG features.
High test accuracy (~99%) is achieved for all of
`YCrCb`, `LUV`, `YUV`, and `HSV` color spaces.  `YCrCb` color space is
selected for this implementation.

The details on sweeping across various parameter space can be found in the
train classifier output logs,
[results/train_classifier.log](results/train_classifier.log). 

---

## Sliding Window Search

### 1. Determine Slide Window Positions
The source code that determines slide window position is in
[py-src/p05_04_determine_slide_window.py](py-src/p05_04_determine_slide_window.py).

In this step, the positions and scales of sliding windows are studied.
Four window sizes, which vary from (64, 64) to (256, 256), are used.
The upper parts (about half) of the images are disregarded because a vehicle
does not appear there.
From the remainder, some lower parts are also disregarded
as the window size reduces because smaller vehicle images do not
appear on the lower parts of the view.
The overlap ratio of 0.75 is used.  In other words, the detection window
moves in a step size of the quarter of the window size.

The image below shows slide window positions for each window size.

![Slide Window Positions][img_slide_win_01]


### 2. Slide Window Vehicle Search
The source code for slide window vehicle search is located in
[py-src/p05_05_slide_window_search.py](py-src/p05_05_slide_window_search.py).

#### Slide Window Search
At each slide window position determined in the previous step, the image
under the window is converted to `YCrCb` color space and
resized to `64x64` scale in order to match the trained classifier.
Then, the trained classifier is used to predict whether the image
represents vehicle pixels or not.

The classifier uses binned spatial features, color histogram, and HOG features
to predict the image classification.

The slide window search results of test images are located under:
[output_images/slide_search/](output_images/slide_search)

An example image is included below.
![Slide Window Search][img_slide_search_01]

#### Heat Map
The search result shown above contains multiple detections on a single
vehicle and also some false positives.  In order to improve the prediction
accuracy, a heat-map technique is used.

The heat-map threshold of 4 is used in this implementation.  In other words,
a pixel is identified as a vehicle only when more than four detections are
found.

Then, `scipy.ndimage.measurements.label()` function is used to determine
the final vehicle bounding boxes from the heat-map.

The heat-map result of each test images is located under:
[output_images/heat_map/](output_images/heat_map)

An example image is shown below.
![Heat Map][img_heat_map_01]

#### Final Vehicle Detection

The final vehicle detected images are located under:
[output_images/vehicle_detection/](output_images/vehicle_detection)

An example image is shown below.
![Vehicle Detection][img_vehicle_detection_01]


#### Performance Enhancement
Extracting HOG feature is the most expensive (time consuming) operation
during the slide window search.  Since overlaps exist among neighboring
slide windows, it would be inefficient to compute HOG feature at each 
slide window.

Instead, the HOG features are extracted for the entire region of interest
at each image scale.
Then, bounding boxes for the slide windows are mapped to the HOG feature
block dimensions.
This technique significantly reduces the number of HOG feature extractions
and therefore improves the pipeline runtime performance.

---

## Vehicle Detection on Video

The techniques used on image detection are equally applicable on video detection.
In other words, the vehicle detection is performed on each frame of video
using Linear SVM classifier and sliding window techniques.

HOG features, binned spatial color features, and color histograms are used
as features to train and predict vehicles from given video frames.

Also, heat-map and `scipy.ndimage.measurements.label()` are used in order to
filter out false-positives and to identify clean bounding box for detected
vehicles.
_(Detailed description of this process is included in the above image
processing section.)_

#### Cross-Frame Optimization
In addition, the heat-map from previous frame is carried over to the next
frame.  The previous heat-map contribution factor of 0.5 is selected
after multiple experiments.  This means that the new heat-map is computed
as an average of heat-map from current frame and contributions from previous
frame.  This further improved the prediction accuracy by rejecting isolated
false-positive instances.

This a link to the project video output:
- [project_video.mp4](./output_images/video/project_video.mp4)

---

## Discussion

### 1. Limitation and Future Works
Major drawback of this implementation is its runtime performance.  It took
over 30 minutes to process 50 seconds project video on my MacBook Pro laptop.
Perhaps, the better performance can be achieved on more powerful devices,
especially ones with GPUs.  However, the runtime performance enhancements
are inevitable in order to perform real-time vehicle detections.

There are several potential approaches to improve the pipeline runtime.
It is possible to further utilize characteristics from previous predictions
to reduce search iterations.  More optimization on search area can also be
performed.

However, there can be trade-offs between runtime performance and accuracy.
More thorough analyses are required to find optimal way of balancing
runtime and prediction accuracy.

---

## Appendix
### 1. Source and Outputs
#### Source Code
- Vehicle Detection 
  - [p05_vehicle_detection_main.py](py-src/p05_vehicle_detection_main.py)
  - [p05_01_correct_distortion.py](py-src/p05_01_correct_distortion.py)
  - [p05_02_feature_extraction.py](py-src/p05_02_feature_extraction.py)
  - [p05_03_train_classifier.py](py-src/p05_03_train_classifier.py)
  - [p05_04_determine_slide_window.py](py-src/p05_04_determine_slide_window.py)
  - [p05_05_slide_window_search.py](py-src/p05_05_slide_window_search.py)
- Misc
  - [my_util.py](py-src/my_util.py)

#### Execution Log
- Classifier Training 
  - [train_classifier.log](results/train_classifier.log)

#### Output Images
- Feature Extraction
  - [output_images/feat_extract/](output_images/feat_extract/)
- Distortion Correction
  - [output_images/undistorted/](output_images/undistorted/)
- Slide Window Location
  - [output_images/slide_win/](output_images/slide_win/)
- Slide Window Search
  - [output_images/slide_search/](output_images/slide_search/)
- Heat Map
  - [output_images/heat_map/](output_images/heat_map/)
- Vehicle Detection
  - [output_images/vehicle_detection/](output_images/vehicle_detection/)
  
#### Output Videos
- Videos
  - [output_images/video/](output_images/video/)

#### Other Output Files
- Camera Calibration Pickle File
  - [camera_cal.p](results/camera_cal.p)
- Trained Classifer Pickle Files
  - [classifier_YCrCb_sp32_hist32_hog_9_8_2_ALL.p](results/classifier_YCrCb_sp32_hist32_hog_9_8_2_ALL.p)
  - [classifier_HLS_sp32_hist32_hog_9_8_2_ALL.p](results/classifier_HLS_sp32_hist32_hog_9_8_2_ALL.p)
  - [classifier_HSV_sp32_hist32_hog_9_8_2_ALL.p](results/classifier_HSV_sp32_hist32_hog_9_8_2_ALL.p)
  - [classifier_LUV_sp32_hist32_hog_9_8_2_ALL.p](results/classifier_LUV_sp32_hist32_hog_9_8_2_ALL.p)
  - [classifier_YCrCb_hist32.p](results/classifier_YCrCb_hist32.p)
  - [classifier_YCrCb_hog_9_8_2_ALL.p](results/classifier_YCrCb_hog_9_8_2_ALL.p)
  - [classifier_YCrCb_sp32_hist32_hog_9_8_2_0.p](results/classifier_YCrCb_sp32_hist32_hog_9_8_2_0.p)
  - [classifier_YCrCb_sp32_hist32_hog_9_8_2_1.p](results/classifier_YCrCb_sp32_hist32_hog_9_8_2_1.p)
  - [classifier_YCrCb_sp32_hist32_hog_9_8_2_2.p](results/classifier_YCrCb_sp32_hist32_hog_9_8_2_2.p)
  - [classifier_YCrCb_sp32.p](results/classifier_YCrCb_sp32.p)
  - [classifier_YUV_sp32_hist32_hog_9_8_2_ALL.p](results/classifier_YUV_sp32_hist32_hog_9_8_2_ALL.p)
