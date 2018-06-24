# Vehicle Detection
All process pictures are in Output_images folder

## Overview
Detect vehicles using HOG + SVM classifier with sliding windows. 

The overall pipeline is the following:

* Gather and organize the data
* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images
* Train a linear SVM classifier on normalized HOG features
* Implement a sliding-window technique and use trained classifier to search for vehicles in images
* Run the above steps on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## Pipeline details
### Gather and organize the data
I downloaded Udacity's data for this project, in which Udacity provided [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) images of size 64x64 pixels. The vehicle and non-vehicle images were extracted from the [GTI](http://www.gti.ssr.upm.es/data/Vehicle_database.html) and [KITTI](http://www.cvlibs.net/datasets/kitti/) datasets.


The following is an example of an image in the "vehicle" class:



The following is an example of an image in the "non-vehicle" class:



### Histogram of Oriented Gradients (HOG)
At the highest level, the code to extract features is in the function `extract_features()` in the file 'features.py'. More specifically, the code to extract HOG features is in the function `get_hog_features()` in the file 'features.py'. It uses scikit-learn's `hog()` function.

To extract the optimal HOG features, I experimented with different color spaces and tweaked different parameters. Via trial-and-error, I iterated through multiple loops of HOG feature parameter tweaking, visualizing static images, visualzing the final effect on video, more HOG feature parameter tweaking, etc.

After much experimentation, I settled on the following HOG feature extraction parameters:

* Color space: RGB
* Channel: (all)
* Orientations: 30
* Pixels per cell: 16
* Cells per block: 2



Below is a visualization of the HOG features on the example vehicle and non-vehicle images.

Vehicle HOG:



Non-vehicle HOG:



Visually we can see that the HOG representation of the vehicle is significantly different than that of the non-vehicle example.



### Train a linear SVM classifier
After the HOG features have been extracted for each training image, I trained a linear SVM classifer using these features. The code to do so is in the function `train()` in the file 'train.py'.

In the function `train()`, I first extract the HOG features via `extract_features()`. Then, I intentionally unbalance the data, such that the non-vehicle to vehicles ratio is 3:1, via the following line of code: `X = np.vstack((car_features, notcar_features, notcar_features, notcar_features)).astype(np.float64)`. I did this because in the video output, I was originally getting too many false positives, so I intentionally unbalanced the data to have more negatives (i.e. non-vehicles).


Using a train/test , I trained the linear SVM, and saved the final trained model to 'car_detection.p'. 


Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 6108
26.21 Seconds to train SVC...
Test Accuracy of SVC =  0.9529
19.68 Seconds to train LRC...
Train Accuracy of LRC= 0.9986
Test Accuracy of LRC= 0.9686
My LRC predicts: [0. 1. 0. 1. 1. 1. 0. 0. 1. 1.]
For these 10 labels: [0. 1. 0. 1. 1. 1. 0. 0. 1. 0.]
0.00087 Seconds to predict 10 labels with LRC
19.42 Seconds to train MLP...
Train Accuracy of MLP= 1.0
Test Accuracy of MLP= 0.9919
My MLP predicts: [0. 1. 0. 1. 1. 1. 0. 0. 1. 0.]
For these 10 labels: [0. 1. 0. 1. 1. 1. 0. 0. 1. 0.]
0.00129 Seconds to predict 10 labels with LRC












### Sliding window search
I implemented a basic sliding window search to detect areas in the image/video where a vehicle is likely present. A window size of 96x96 pixels worked well for the scale of vehicles present in the project video, so I only used this window size for this project. A potential enhancement is to use multiple window sizes across different areas of the image, e.g. smaller window sizes for areas closer to the horizon.

Since we know cars will not be present in the sky, and unlikely to be present in full view near the side edges of the image, I limited the search area of my sliding window search . This allows the vehicle detector to perform faster than searching through the entire image, and also reduces potential false positives in areas like the sky and trees.

I chose an overlap percentage of 0.7 (i.e. 70%). I found this gave me reliable detections with multiple detected bounding boxes on vehicles, i.e. the density of detections around vehicles is high, relative to non-vehicle areas of the image. This allows me to create a high-confidence prediction of vehicles in my heat map (more details later) and reject potential false positives.

The code to perform my sliding window search is in the functions `slide_window()` and `search_windows()` in the file 'windows.py'. These functions serve the same purpose as those presented in the lecture notes, where `slide_window()` returns a list of windows to search, and `search_windows()` uses the pre-trained HOG+SVM classifier to classify the image in each window as either "vehicle" or "non-vehicle". These function are called in the function `annotate_image()` in 'detect_video.py', and we can see the parameters passed to `slide_window()` in the function call. Note `pct_overlap` is the percentage overlap parameter, and it is defined in the file 'settings.py'.

Below is an example of running my sliding window search on an image (blue boxes indicate a vehicle detection in that window):


We can see there are many boxes detected as vehicles, even though not all boxes are vehicles.

### Final bounding box prediction
As seen previously, there are many boxes detected as vehicles, but many of those boxes do not actually enclose a vehicle. However, notice that the density of boxes tend to be high around actual vehicles, so we can take advantage of this fact when predicting the final bounding box. We can do so via a heat map. I created a heat map by adding the contributions of each predicted bounding box, similar to the method presented in the lectures. A heat map created from the previous image is as follows:



After a heat map is generated, we threshold the heatmap into a binary image, then use scikit-learn's `label()` function to draw the final bounding boxes based on our thresholded heat map. The heatmap threshold is specified by the variable `heatmap_thresh` in the file 'settings.py' (for a static image like this one, I used a `heatmap_thresh` of 1). Using this method, the final bounding boxes are predicted as such:


The above illustrations were based on a static image. However, in a video stream, we can take advantage of the temporal correlation between video frames. I can reduce the number of false positives by keeping track of a cumulative heat map over the past 30 frames in a video, and threshold the cumulative heatmap. The cumulative heatmap is enabled by a queue of "hot windows".



## Final video output

## Discussion
The main challenge for this project was parameter tuning, mostly to reduce the number of false positives in the video. Even though the HOG+SVM classifier reported good results after training, it did not necessarily mean good results in the overall vehicle detection task.



