from util import *
from processors import *

import argparse
import cv2
from math import pi
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import numpy as np

# TODO: Figure out why not working on test6.jpg
# TODO: Filtering, hints

def valid(left_line, right_line):
    '''
    Check whether a pair of left and right lane lines are a valid
    detection
    '''
    # Check that both lines detected
    if not left_line.detected or not right_line.detected:
        print("Both lines not detected")
        return False
    # Check curvature similarity
    width = abs(left.radius() - right.radius())
    if width < 2.0 or width > 5.0:
        print("Radii not similar (width={})".format(width))
        return False
    # Check lines parallel
    # by checking the variance of the widths at many points
    y_vals = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]
    widths = left_line.vals_m(y_vals) - right_line.vals_m(y_vals)
    width_var = np.var(widths)
    if width_var > 0.2: # TODO: Tune this
        print("Lines not parallel (width variance={})".format(width_var))
        return False
    # Check curvature is sane
    # See http://onlinemanuals.txdot.gov/txdotmanuals/rdw/horizontal_alignment.htm
    mean_curvature = (2 * left.curvature() * right.curvature()) / (left.curvature() + right.curvature())
    if mean_curvature > 0.005679: # Curvature in 1/m for radius = 587 ft
        print("Curvature is too large (curvature={})".format(mean_curvature))
        return False
    return True

def process(img, undistorter, lane_extractor, transformer, lane_fitter, last_left=None, last_right=None, show_all=False):
    '''
    Full processing pipeline for lane lines

    img -- The input image, directly from the camera
    undistorter -- An Undistorter processor object
    lane_extractor -- A LaneExtractor processor object
    transformer -- A GroundProjector processor object
    lane_fitter -- A LaneFitter processor object
    last_left -- (optional) A Line object representing the left lane line in the
        previous frame of video
    last_right -- (optional) A Line object representing the right lane line in
        the previous frame of video
    show_all -- If True, show all intermediate images in PyPlot figures
        (default False)

    Returns (composite_img, (left_lane, right_lane)

    composite_img -- The input image, undistorted, with the lane drawn on it in
        green
    left_lane -- A Line object representing the left lane line
    right_lane -- A Line object representing the right lane line
    '''
    try:
        undistorted = undistorter.undistortImage(img)
        lane_img = lane_extractor.extract_lanes(undistorted)
        transformed_lane_img = transformer.transformImage(lane_img)
        (left_lane, right_lane) = lane_fitter.fit_lanes(transformed_lane_img, last_left, last_right)
        curvature_img = draw_lane(left_lane, right_lane, transformed_lane_img.shape, lane_fitter.resolution)
        curvature_img_warped = transformer.inverseTransformImage(curvature_img, undistorted.shape)
        # TODO: Validity check
        composite_img = cv2.addWeighted(undistorted, 1, curvature_img_warped, 0.3, 0)
        if show_all:
            polyfit_img = plot_on_img(transformed_lane_img, left_lane, right_lane, color='yellow')
            return (img,
                    undistorted,
                    lane_img,
                    transformed_lane_img,
                    (left_lane, right_lane),
                    polyfit_img,
                    curvature_img,
                    curvature_img_warped,
                    composite_img)
        return composite_img, (left_lane, right_lane)
    except Exception as e:
        # TODO: This is a quick hack
        print("Exception: {}".format(e))
        return undistorted, (Line(), Line())
def main():
    parser = argparse.ArgumentParser(description='Process an image to find lane lines')
    parser.add_argument('--camera-matrix', type=str,
                        help='File path of camera matrix, stored as .npy',
                        default='camera_matrix.npy')
    parser.add_argument('--distortion-coefficients', type=str,
                        help='File path of camera distortion coefficients, stored as .npy',
                        default='disortion_coefficients.npy')
    parser.add_argument('--resolution', type=int,
                        help='Resolution (px/meter) for top-down images',
                        default=200)
    parser.add_argument('input_file', type=str,
                        help='File path of the image/video to process')
    parser.add_argument('output_file', type=str,
                        help="File path to store the output", nargs='?')
    args = parser.parse_args()
    K = np.load(args.camera_matrix)
    D = np.load(args.distortion_coefficients)
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

    input_ext = args.input_file[-3:]
    if input_ext in ['jpg', 'png']:
        input_img = mpimg.imread(args.input_file)
        composite_img, (left_lane, right_lane) = process(input_img, undistorter, lane_extractor, transformer, lane_fitter)
        plt.figure()
        plt.imshow(composite_img)
        plt.show()
        if args.output_file:
            print("Saving file to {}".format(args.output_file))
            mpimg.imsave(args.output_file, composite_img)
    elif input_ext in ['mp4']:
        process_frame = lambda frame: process(frame, undistorter, lane_extractor, transformer, lane_fitter)[0]
        clip = VideoFileClip(args.input_file)
        clip = clip.fl_image(process_frame)
        print("Writing video file to {}".format(args.output_file))
        clip.write_videofile(args.output_file, audio=False)
    else:
        print("Invalid input file extension .{}".format(input_ext))
if __name__ == "__main__":
    main()
