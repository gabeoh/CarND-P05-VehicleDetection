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
* Histogram of Oriented Gradients (HOG)
* Binned color features
* Histograms of color 

**Explain how (and identify where in your code) you extracted HOG features 
from the training images.**

_Explanation given for methods used to extract HOG features, including 
which color space was chosen, which HOG parameters (orientations, 
pixels_per_cell, cells_per_block), and why._

The code for this step is contained in the first code cell of the IPython 
notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  
Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` 
parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  
I grabbed random images from each of the two classes and displayed them to 
get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of 
`orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]


**Explain how you settled on your final choice of HOG parameters.**

I tried various combinations of parameters and...


---
## Classifier Training

**Describe how (and identify where in your code) you trained a classifier 
using your selected HOG features (and color features if you used them).**

I trained a linear SVM using...


---
## Sliding Window Search

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

