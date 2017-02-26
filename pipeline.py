from util import *
from processors import *

import argparse
import cv2
from math import pi
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import numpy as np
import pickle

# TODO: Figure out why not working on test6.jpg
# TODO: Filtering, hints

def valid(left_line, right_line):
    '''
    Check whether a pair of left and right lane lines are a valid
    detection
    '''
    # Check that both lines detected
    if left_line is None or right_line is None:
        print("Both lines not detected")
        return False
    # Check vehicle is in lane
    veh_pos = (right_line.dist_from_center_m() + left_line.dist_from_center_m())/2
    if abs(veh_pos) > 1.0:
        print("Vehicle is not in the center of the lane (position={})".format(veh_pos))
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
    mean_curvature = 2.0/(left_line.radius() + right_line.radius())
    if mean_curvature > 0.005679: # Curvature in 1/m for radius = 587 ft
        print("Curvature is too large (curvature={})".format(mean_curvature))
        return False
    return True


class Pipeline(object):
    def __init__(self, undistorter, lane_extractor, transformer, lane_fitter, show_all=False):
        '''
        Full processing pipeline for lane lines
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
        '''
        self.undistorter = undistorter
        self.lane_extractor = lane_extractor
        self.transformer = transformer
        self.lane_fitter = lane_fitter
        self.show_all = show_all
        self.left_lane = None
        self.right_lane = None
        self.max_age = 5

    def __call__(self, img):
        '''
        Process a frame for lane lines

        img -- The input image, directly from the camera

        Returns composite_img

        composite_img -- The input image, undistorted, with the lane drawn on it in
            green
        '''
        try:
            undistorted = self.undistorter.undistortImage(img)
            lane_img = self.lane_extractor.extract_lanes(undistorted, show_plots = self.show_all)
            transformed_lane_img = self.transformer.transformImage(lane_img)
            if self.show_all:
                plt.figure()
                plt.title('Top-down Binary Lane Image')
                plt.imshow(transformed_lane_img, cmap='gray')
            (left_lane, right_lane) = self.lane_fitter.fit_lanes(transformed_lane_img, self.left_lane, self.right_lane)
            for line in left_lane, right_lane:
                line.middle_x = self.transformer.transformPoint([undistorted.shape[1]/2, undistorted.shape[0]])[0][0][0]
                line.closest_y = transformed_lane_img.shape[0]
            if valid(left_lane, right_lane):
                self.left_lane = left_lane
                self.right_lane = right_lane
            else:
                if self.left_lane is None:
                    self.left_lane = left_lane
                if self.right_lane is None:
                    self.right_lane = right_lane
            curvature_img = draw_lane(self.left_lane, self.right_lane, transformed_lane_img.shape, self.lane_fitter.resolution)
            curvature_img_warped = self.transformer.inverseTransformImage(curvature_img, undistorted.shape)
            composite_img = cv2.addWeighted(undistorted, 1, curvature_img_warped, 0.3, 0)
            veh_position = (self.right_lane.dist_from_center_m() + self.left_lane.dist_from_center_m())/2
            curvature = self.left_lane.curvature()
            info = "Position:  {:.3f} m".format(veh_position)
            text_position = (10, 50)
            composite_img = cv2.putText(composite_img, info, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
            text_position = (10, 90)
            info = "Curvature: {:.6f} 1/m".format(curvature)
            composite_img = cv2.putText(composite_img, info, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
            return composite_img
        except Exception as e:
            # TODO: This is a quick hack
            print("Exception: {}".format(e))
            return undistorted


def main():
    parser = argparse.ArgumentParser(description='Process an image to find lane lines')
    parser.add_argument('--camera-matrix', type=str,
                        help='File path of camera matrix, stored as .npy',
                        default='camera_matrix.npy')
    parser.add_argument('--distortion-coefficients', type=str,
                        help='File path of camera distortion coefficients, stored as .npy',
                        default='disortion_coefficients.npy')
    parser.add_argument('--projector', type=str,
                        help='Pickled projector to use',
                        default='projector.p')
    parser.add_argument('--show-all', action='store_true',
                        help='Show all intermediate images')
    parser.add_argument('--subclip', type=float, nargs=2, required=False,
                        help='Beginning and end times of video')
    parser.add_argument('input_file', type=str,
                        help='File path of the image/video to process')
    parser.add_argument('output_file', type=str,
                        help="File path to store the output", nargs='?')
    args = parser.parse_args()

    # Load the undistorter from files
    K = np.load(args.camera_matrix)
    D = np.load(args.distortion_coefficients)
    undistorter = Undistorter(K, D)

    # Load the projector/transformer from file
    with open(args.projector, 'rb') as f:
        transformer = pickle.load(f)

    # Set up the binary lane extractor
    lane_extractor = LaneExtractor(0.9)
    
    # Set up the lane fitter
    lane_fitter = LaneFitter(transformer.getResolution())

    # Create the processing pipeline
    process = Pipeline(undistorter, lane_extractor, transformer, lane_fitter, args.show_all)

    # Load and process the file
    input_ext = args.input_file[-3:]
    if input_ext in ['jpg', 'png']:
        input_img = mpimg.imread(args.input_file)
        composite_img = process(input_img)
        if args.output_file:
            print("Saving file to {}".format(args.output_file))
            composite_img = cv2.cvtColor(composite_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(args.output_file, composite_img)
        else:
            plt.figure()
            plt.imshow(composite_img)
            plt.show()
    elif input_ext in ['mp4']:
        clip = VideoFileClip(args.input_file)
        if args.subclip and len(args.subclip) == 2:
            clip = clip.subclip(args.subclip[0], args.subclip[1])
        clip = clip.fl_image(process)
        print("Writing video file to {}".format(args.output_file))
        clip.write_videofile(args.output_file, audio=False)
    else:
        print("Invalid input file extension .{}".format(input_ext))
if __name__ == "__main__":
    main()
