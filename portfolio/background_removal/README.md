# Using Gaussian Mixture Models to Isolate Movement in Video

### Experiments in removing the background from video from scratch

## Description

---

OpenCV has an algorithm `createBackgroundSubtractorMOG2` that allows you to create a background model from a video which you can then use to create a mask that isolates just the moving parts of that same video.

Under the hood, it uses Gaussian Mixture Models (GMMs) to build up a distribution of color values at each pixel for a subsample of frames (this can be adjusted withe the `history` parameter of the method)

To better understand this algorithm and Gaussian Mixture Models better, I attempt remove the background from a short clip of traffic moving down a highway.

<img src="./portfolio/background_removal/assets/traffic.gif" width=300 />

## Methods

---

To remove the background, I first create a model of it by initializing a GMM for each pixel in a the frame of a video. This is a relitivly short video so I will not be subsampling frames to build the history that represents the background, rather I will be using all the frames.

Each GMM is fit to the RGB pixel values as they change through each frame of the video. We can then detect anomolies be scoring new pixel values with each model. For example, below you can see every RGB value for a specific pixel scored with the associated GMM. Very yellow points are outliers and will be considered bright and moving in the frame they appear in

<img src="./portfolio/background_removal/assets/scatter1.png" width=300 />

All the points in a single frame can be evaluated in the same way, which yeilds a result like below, where yellow areas indicate movement and purple areas indicate background

<img src="./portfolio/background_removal/assets/frame1.png" width=300 />

## Results

---
### From Scratch Implementation:

<img src="./portfolio/background_removal/assets/traffic_movement_scratch.gif" width=300 />

---
### OpenCV algorithm:

<img src="./portfolio/background_removal/assets/traffic_movement.gif" width=300 />

While I am impressed by the results I was able to achieve from scratch, the OpenCV  implementation has some distinct advantages:
- It is much faster, the 8 second video was processed in less than a second. While I'm sure speed is dependent on frame size and resolution, the OpenCV algorithm could be used for real-time applications.
- Can work on-line, the OpenCV implementation allows you to build up a background model frame by frame as they become available. Again suited well for real-time applications.
- Allows you to specify a history length. Old frames are thrown out and removed form the model after a certain number of frames have accumulated. This has the effect of ignoring, after some time, parts of a video that were once moving but then became stationary.
- OpenCV implementation can differentiat shadows and can remove them entirely if desired

There appears to be artifacts that flash in front of the entire video of the OpenCV implementation that are not preseent on the scratch implementation. These could be reduced by increasing the threshold value passed into `createBackgroundSubtractorMOG2`

One fun thing that I can do with my scratch implementation is take the average pixel value from each GMM and build up an image of just the background, excluding all moving parts of the video.

<img src="./portfolio/background_removal/assets/bg.png" width=300>

Even though my approach won't be used in production anytime soon, it was very interesting and educational to dive into this algorithm more in-depth and use GMMs in a way that I would not have otherwise considered.

