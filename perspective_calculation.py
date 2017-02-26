import cv2
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from processors import GroundProjector
from util import comparison_plot

'''
Script to generate the perspective projector and save it to a Python pickle.
'''

# These points were chosen by hand to be a rectangle with points on
# lane markers, which should be 120 feet apart.
img_pts = np.float32([(590, 456),  # Far left
                      (464, 548),  # Near left
                      (831, 548),  # Near right
                      (694, 456)]) # Far right
# How much to include on either side of the lane
offset = 2.0 # meters
# How much to include past the end of the chosen lane marker
top_offset = 3.0
# How much to include before the beginning of the chosen lane marker.
# This should be about at the edge of the hood of the car
bottom_offset = 7.0
# Width of the lane
lane_width = 3.7 # meters
# Length of 3 lane markers
line_length = 3 * 40.0*0.30 # Feet to meters
# Pixels per meeter (0.5 cm resolution)
px_m = 200.0
# Calculate the points' positions on the ground in meters
obj_pts = [(offset, top_offset),                           # Far left
           (offset, top_offset + line_length),             # Near Left
           (offset+lane_width, top_offset + line_length),  # Near right
           (offset+lane_width, top_offset)]                # Far right
# Convert from meters to pixels on the ground
obj_pts = np.float32(obj_pts) * px_m
# Calculate the perspective matrix
P = cv2.getPerspectiveTransform(img_pts, obj_pts)
# Calculate the shape of the output image
output_shape = (int(px_m * (offset * 2 + lane_width)),
                int(px_m * (top_offset + bottom_offset + line_length)))
# Create a projector object
projector = GroundProjector(P, px_m, output_shape)

# Print results
print("Projection matrix:")
print(P)
print("Pixels per meter: {}".format(px_m))
print("Output shape (pixels): {}".format(output_shape))

# Draw results
transformed_pts = projector.transformPoint(img_pts).astype(np.int32)
img_pts = img_pts.astype(np.int32)
filename = 'test_images/straight_lines1.jpg'
img = mpimg.imread(filename)
warped_img = projector.transformImage(img)
cv2.polylines(img, [img_pts], isClosed=True, color=(255, 0, 0), thickness=3)
cv2.polylines(warped_img, transformed_pts, isClosed=True, color=(255, 0, 0), thickness=int(px_m/10))
comparison_plot(img, warped_img, 'Original', 'Warped', filename)
plt.show()

# Save results
print("Saving the projector to projector.p")
with open('projector.p', 'wb') as f:
    pickle.dump(projector, f)
