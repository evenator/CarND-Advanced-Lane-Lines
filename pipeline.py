from util import *
from processors import *

import argparse
import cv2
from math import pi
import matplotlib.image as mpimg
import numpy as np

# TODO: Figure out why not working on test3.jpg
# TODO: Figure out why not working on test6.jpg
# TODO: Make this work on straightlines1.jpg (use joint fit)


def valid(left_line, right_line):
    # Check that both lines detected
    if not left_line.detected or not right_line.detected:
        print("Both lines not detected")
        return False
    # Check curvature similarity
    width = abs(1.0/left_curvature - 1.0/right_curvature)
    if width < 2.0 or width > 5.0:
        print("Curvatures not similar (width={})".format(width))
        return False
    # Check lines parallel
    # by checking the variance of the widths at many points
    y_vals = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]
    widths = left_line.vals(y_vals) - right_line.vals(y_vals)
    width_var = np.var(widths)
    if width_var > 0.2: # TODO: Tune this
        print("Lines not parallel (width variance={})".format(width_var))
        return False
    # Check curvature is sane
    # See http://onlinemanuals.txdot.gov/txdotmanuals/rdw/horizontal_alignment.htm
    mean_curvature = (2 * left.curvature * right.curvature) / (left.curvature + right.curvature)
    if mean_curvature > 0.005679: # Curvature for radius = 587 ft
        print("Curvature is too large (curvature={})".format(mean_curvature))
        return False
    return True

def process(img, undistorter, lane_extractor, transformer, lane_fitter, last_left=None, last_right=None, output_all=False):
    undistorted = undistorter.undistortImage(img)
    lane_img = lane_extractor.extract_lanes(undistorted)
    transformed_lane_img = transformer.transformImage(lane_img)
    (left_lane, right_lane) = lane_fitter.fit_lanes(transformed_lane_img, last_left, last_right)
    polyfit_img = plot_on_img(transformed_lane_img, left_lane.poly, right_lane.poly, color='yellow')
    curvature_img = draw_lane(left_lane, right_lane, transformed_lane_img.shape, lane_fitter.resolution)
    curvature_img_warped = transformer.inverseTransformImage(curvature_img, undistorted.shape)
    # TODO: Validity check
    composite_img = cv2.addWeighted(undistorted, 1, curvature_img_warped, 0.3, 0)
    if output_all:
        return (img, 
                undistorted,
                lane_img,
                transformed_lane_img,
                (left_lane, right_lane),
                polyfit_img,
                curvature_img,
                curvature_img_warped,
                composite_img)
    else:
        return composite_img, (left_lane, right_lane)

def main():
    # TODO: UI
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera-matrix', type=str,
                        help='File path of camera matrix, stored as .npy',
                        default='camera_matrix.npy')
    parser.add_argument('--distortion-coefficients', type=str,
                        help='File path of camera distortion coefficients, stored as .npy',
                        default='disortion_coefficients.npy')
    parser.add_argument('--resolution', type=int,
                        help='Resolution (px/meter) for top-down images',
                        default=200)
    parser.add_argument('image', type=str,
                        help='File path of the image to process')
    args = parser.parse_args()
    K = np.load(args.camera_matrix)
    D = np.load(args.distortion_coefficients)
    input_img = mpimg.imread(args.image)
    undistorter = Undistorter(K, D)
    # TODO: Document these numbers
    P = np.array([[ -6.16890178e-01,  -1.79811526e+00,   1.14922653e+03],
                  [ -3.80945275e-15,  -2.53733237e+01,   1.15182167e+04],
                  [ -6.27651414e-19,  -2.38310982e-03,   1.00000000e+00]])
    # TODO: Make ground image shape configurable
    ground_img_shape = (1540, 9200) # pixels = (7.7, 46) meters
    transformer = GroundProjector(P, ground_img_shape)
    # TODO: Tune lane extractor
    lane_extractor = LaneExtractor(25, (0, pi/3), (30, 100))
    lane_fitter = LaneFitter(args.resolution)
    # TODO: Video pipeline, filtering
    composite_img, (left_lane, right_lane) = process(input_img, undistorter, lane_extractor, transformer, lane_fitter)
    plt.figure()
    plt.imshow(composite_img)
    plt.show()

if __name__ == "__main__":
    main()
