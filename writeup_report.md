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

[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4


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

First, the positions of slide windows are determined.
Five window sizes, which vary from (50, 50) to (320, 280), are used.
The upper parts (about half) of the images are disregarded because a vehicle
does not appear there.
From the remainder, some lower parts are also disregarded
as the window size reduces because smaller vehicle images do not
appear on the lower parts of the view.

For bigger windows, the overlap ratio of 0.75 is used.  For the smallest
two windows, the ratio is reduced to 0.5 because even the reduced overlap
yields enough granularity for the smaller window sizes.

The image below shows slide window positions for each window size.

![Slide Window Positions][img_slide_win_01]


### 2. Vehicle Detection


**Describe how (and identify where in your code) you implemented a sliding 
window search.  How did you decide what scales to search and how much to 
overlap windows?**

_A sliding window approach has been implemented, where overlapping tiles in 
each test image are classified as vehicle or non-vehicle. Some justification 
has been given for the particular implementation chosen._

I decided to search random window positions at random scales all over the 
image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]



**Show some examples of test images to demonstrate how your pipeline is 
working. How did you optimize the performance of your classifier?**

_Some discussion is given around how you improved the reliability of the 
classifier i.e., fewer false positives and more reliable car detections 
(this could be things like choice of feature vector, thresholding the 
decision function, hard negative mining etc.)_

Ultimately I searched on two scales using YCrCb 3-channel HOG features 
plus spatially binned color and histograms of color in the feature vector, 
which provided a nice result.  Here are some example images:

![alt text][image4]


---
## Vehicle Detection on Video

**Provide a link to your final video output.  Your pipeline should perform 
reasonably well on the entire project video (somewhat wobbly or unstable 
bounding boxes are ok as long as you are identifying the vehicles most of 
the time with minimal false positives.)**

_The sliding-window search plus classifier has been used to search for and 
identify vehicles in the videos provided. Video output has been generated 
with detected vehicle positions drawn (bounding boxes, circles, cubes, etc.) 
on each frame of video._

Here's a [link to my video result](./project_video.mp4)


**Describe how (and identify where in your code) you implemented some kind 
of filter for false positives and some method for combining overlapping 
bounding boxes.**

_A method, such as requiring that a detection be found at or near the same 
position in several subsequent frames, (could be a heat map showing the 
location of repeat detections) is implemented as a means of rejecting 
false positives, and this demonstrably reduces the number of false positives. 
Same or similar method used to draw bounding boxes (or circles, cubes, etc.) 
around high-confidence detections where multiple overlapping detections occur._

I recorded the positions of positive detections in each frame of the video.  
From the positive detections I created a heatmap and then thresholded that 
map to identify vehicle positions.  I then used 
`scipy.ndimage.measurements.label()` to identify individual blobs in 
the heatmap.  I then assumed each blob corresponded to a vehicle.  
I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of 
video, the result of `scipy.ndimage.measurements.label()` and the bounding 
boxes then overlaid on the last frame of video:

Here are six frames and their corresponding heatmaps:

![alt text][image5]

Here is the output of `scipy.ndimage.measurements.label()` on the 
integrated heatmap from all six frames:
![alt text][image6]

Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]


---
## Suggestions to Make Your Project Stand Out!
**A stand out submission for this project will be a pipeline that runs in 
near real time (at least several frames per second on a good laptop) and 
does a great job of identifying and tracking vehicles in the frame with a 
minimum of false positives. As an optional challenge, combine this vehicle 
detection pipeline with the lane finding implementation from the last project! 
As an additional optional challenge, record your own video and run your 
pipeline on it to detect vehicles under different conditions.**


---
## Discussion

### 1. Limitation and Future Works

**Briefly discuss any problems / issues you faced in your implementation 
of this project.  Where will your pipeline likely fail?  What could you do 
to make it more robust?**

_Discussion includes some consideration of problems/issues faced, what could 
be improved about their algorithm/pipeline, and what hypothetical cases would 
cause their pipeline to fail._

Here I'll talk about the approach I took, what techniques I used, what worked 
and why, where the pipeline might fail and how I might improve it if I were 
going to pursue this project further.  


---
## Appendix
### 1. Source and Outputs
#### Source Code

#### Execution Log

#### Output Images

#### Output Videos

