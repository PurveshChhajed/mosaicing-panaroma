# Mosaicing-(panorama)
## About
A single-clicked image has a limited field of view (FOV), we need to stitch together several image stills to form a mosaic to increase the FOV. Image mosaicing is a very popular way to obtain a wide FOV image of a scene. The basic idea is to capture images as a camera moves and stitch these images together to obtain a single larger image.These multiple image slices can be mosaiced together to give an entire view of a scene.

## Major Pakages and Inbuild functions required to run the code
* OpenCV
* Numpy
* [PerspectiveTransform](https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html)
* [FindHomography](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html?highlight=find%20homography#cv2.findHomography)
* [WarpPerspective](https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html)
* [ORB_create](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html)
* [BFMatcher.knnMatch](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html)

## Approach
Below is a higher-level overview of the mosaicing technique implemented:

* Choose first two images in the dataset
* Find matching keypoints(features) between the two images
* Calculate the homography using the matched key- points
* Using this homography, one image is warped to be in the same frame as the other and a new image of all black pixels is created which can fit both images in the new frame
* Repeat step 2 with the current mosaic and the next image, until all the images in the dataset are covered

## Final Output:
Panorama 1
![Panorama 1](https://github.com/PurveshChhajed/mosaicing-panaroma/blob/master/FinalMosaic_carmel.jpg)
Panorama 2
![Panorama 2](https://github.com/PurveshChhajed/mosaicing-panaroma/blob/master/FinalMosaic_goldengate.jpg)
Panorama 3
![Panorama 3](https://github.com/PurveshChhajed/mosaicing-panaroma/blob/master/FinalMosaic_rio.jpg)

## Conclusions
* The quality of mosaic is deteriorating with increase in the number of stitched images(>15) and which is probably because of the homography error accumulates with each stitch.
* Code giving best result when there is more than 75% of overlap and proper alignment between the consecutive images.
* Need to implement Sequential Bundle Adjustment to reduce the propagating error for better results.
