Self Driving Car Nanodegree Project 4: Advanced Lane Lines
===========================================================

In this project, I used more advanced image processing techniques to detect
the lane in front of a car from images taken by a forward-facing camera. The
general process for this project is as follows:

1. Calibrate the camera using a given set of chessboard images.
2. Use the calculated camera calibration to perform distortion correction on
  the input image.
3. Use color transformations and gradients to create a thresholded binary
  image that emphasizes lane lines.
4. Apply a perspective transformation to rectify the binary image into a top-
  down view.
5. Detect which of the binary pixels are part of the left and right lane lines.
6. Fit a polynomial to the two lane lines.
7. Determine the curvature of the lane and the vehicle position with respect
  to the center of the lane.
8. Draw the detected lane and warp it back onto the original image perspective.
9. Output a vision display of the lane boundaries overlayed on the original
  image, as well as a numerical estimate of the lane curvature and vehicle
  position.

Camera Calibration and Distortion Correction
--------------------------------------------

**Side-by-side comparison showing distortion correction of one of the
 checkerboard images**

**Description**

Binary Lane Image
-----------------

** Side-by-side comparison showing generation of binary lane image**

**Description**

Top-down Perspective Transform
------------------------------

**Side-by-side perspective showing point correspondences**

**Description of transformation method**


** Binary lane image in top-down view**

Lane-line Fitting
-----------------

** Colorized image showing left and right lane pixels and fit**

** Description of pixel selection method**

** Description of fit method**

Vehicle Position and Lane Curvature
-----------------------------------

** Composite image**

** Description**

Video Pipeline
--------------

** Link to video **

** Description (how video pipeline differs from single image processing)**

Discussion
----------

** Discussion of problems/issues faced**

** Room for improvement, including hypothetical cases in which this pipeline fails**
