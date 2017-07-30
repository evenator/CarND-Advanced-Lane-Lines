from .datatypes import FilteredLine as Line

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import lstsq as solve_least_squares
from scipy.signal import find_peaks_cwt


def scaled_abs(img):
    '''
    Take the absolute value of an image, scale it to [0,255] and cast it to uint8
    '''
    abs_img = np.abs(img)
    return np.uint8(255 * abs_img / np.max(abs_img))


def XSobel(img, ksize=3):
    '''
    Perform Sobel gradient along the X axis
    '''
    return cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)


def YSobel(img, ksize=3):
    '''
    Perform Sobel gradient along the Y axis
    '''
    return cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)


class GroundProjector(object):
    '''
    Processor for transforming images from camera perspective to top-down view
    and vice-versa, using perspective transform.
    '''
    def __init__(self, P, resolution, output_size):
        '''
        Create a GroundProjector from a Perspective matrix

        P -- Perspective transform matrix, as created by
            cv2.getPerspectiveTransform
        output_size -- Default shape of the output image **in pixels** as a
            tuple (width, height).
        resolution -- Resolution in pixels per meter.
        '''
        self._P = P
        self._output_size = output_size
        self._resolution = resolution

    def getResolution(self):
        return self._resolution

    def transformImage(self, src, output_size=None):
        '''
        Transform an image from camera perspective to top-down view

        src -- Image to transform, as a numpy Array-like
        output_size -- Shape of the output image **in pixels** as a tuple
            (width, height). If set to None, use the default output shape of
            the GroundProjector. If there is no default output shape, use the
            shape of the src image.
        '''
        if output_size is None:
            output_size = self._output_size
        if output_size is None:
            output_size = (src.shape[0], src.shape[1])
        return cv2.warpPerspective(src, self._P, output_size)

    def transformPoint(self, src):
        '''
        Transform a point from camera perspective to top-down view

        src -- Point to transform, as a numpy Array-like
        '''
        src = np.array(src)
        while len(src.shape) < 3:
            src = np.array([src])
        dst = cv2.perspectiveTransform(src, self._P)
        return dst

    def inverseTransformImage(self, src, output_shape):
        '''
        Transform an image from top-down view to camera perspective

        src -- Image to transform, as a numpy Array-like
        output_shape -- Shape of the output image **in pixels** as a tuple
            (width, height).
        '''
        return cv2.warpPerspective(src, self._P, output_shape[1::-1], flags=cv2.WARP_INVERSE_MAP)

    @classmethod
    def from_point_correspondence(cls, image_pts, object_points, output_resolution,
                                  output_size=None):
        '''
        Create a GroundProjector from image/point correspondences

        image_pts -- At least 4 pixel coordinates in image space
        object_pts -- At least 4 real-world object coordinates corresponding to
            the points in image_pts, with coordinates in meters.
        output_resolution -- The resolution, in pixels/meter, of the output
            image
        output_size -- Default shape of the output image **in meters** as a
            tuple (width, height). If set to None, there is no default output
            size
        '''
        object_points = np.float32(object_points) * output_resolution
        P = cv2.getPerspectiveTransform(image_pts, object_points)
        output_size = (int(output_resolution * output_size[0]),
                       int(output_resolution * output_size[1]))
        return cls(P, output_size)


class Undistorter(object):
    '''
    Processor to perform dewarping using OpenCV's undistort function
    '''
    def __init__(self, K, D):
        '''
        Initialize the processor

        K -- Camera matrix (2-D matrix of 3x3 floats)
        D -- Distortion coefficients (1-D vector of 5 floats)
        '''
        self._K = K
        self._D = D

    def undistortImage(self, src):
        '''
        Undistort the raw camera image
        '''
        return cv2.undistort(src, self._K, self._D)


class LaneExtractor(object):
    '''
    Processor to take in a color (RGB) image in camera perspective and return
    a binary image in camera perspective that contains mostly lane lines
    '''
    def __init__(self, ys_thresh):
        '''
        Constructor

        ys_thresh -- Threshold for lines in the Y+S image, which is scaled to
            [0.0, 1.0]. Recommended value: 0.9
        '''
        self._ys_thresh = ys_thresh

    def extract_lanes(self, img, show_plots=False):
        '''
        Process an RGB image and return a binary image

        img -- The RGB image to process (in camera perspective)
        show_plots -- Show intermediate images in PyPlot figures (default False)

        Returns a binary image in the camera perspective
        '''
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        y = yuv[:, :, 0]
        y = y / np.max(y)
        s = hls[:, :, 2]
        s = s / np.max(s)
        ys = y + s
        ys_uint8 = (ys / np.max(ys) * 255).astype(np.uint8)
        ys_norm = cv2.equalizeHist(ys_uint8).astype(np.float32) / 255
        binary = np.uint8(ys_norm > self._ys_thresh)
        if show_plots:
            f, (a1, a2, a3, a4) = plt.subplots(1, 4)
            f.suptitle('Lane Extraction')
            a1.imshow(y, cmap='gray')
            a1.set_title('Y')
            a2.imshow(s, cmap='gray')
            a2.set_title('S')
            a3.imshow(ys, cmap='gray')
            a3.set_title('Y+S')
            a4.imshow(binary, cmap='gray')
            a4.set_title('Binary')
        return binary


class LaneFitter(object):
    '''
    A Processor that takes in a binary lane image in top-down perspective and
    finds the lane lines.
    '''
    def __init__(self, resolution, search_box_size_margin=None, max_range=30.0):
        '''
        Constructor

        resolution -- Resolution of the input image in meters/pixel
        search_box_size_margin -- Number of pixels on either side of the center
            of the line to search for line pixels
        max_range -- Maximum range from the bottom of the image to include
            pixels (meters)
        '''
        self.resolution = resolution
        if search_box_size_margin is None:
            self.search_box_size_margin = resolution
        else:
            self.search_box_size_margin = search_box_size_margin
        self.search_box_height = int(resolution/2)
        self.recenter_thresh = 1000
        self.max_range = int(max_range * resolution)

    def close_img(self, img):
        '''
        Perform a morphological closing on the image to join clusters of
        separate pixels in blobs.
        '''
        kernel_size = int(self.resolution/2)
        morph_kernel = np.ones((int(self.resolution/2), int(self.resolution/4)))
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, morph_kernel)

    def find_lane_points(self, img):
        '''
        Find both lane line points using a histogram peaks and sliding filter.

        img -- Image to search in
        '''
        height = img.shape[0]
        smoothed_histogram = self.smoothed_histogram(img[int(height/2):, :])
        sorted_peaks = self.find_peaks(smoothed_histogram)
        line1 = sorted_peaks[0]
        for line2 in sorted_peaks[1:]:
            # Ensure that the second line is at least 2 meters away from the first one
            if abs(line2 - line1)/float(self.resolution) > 2.0:
                break
        if line1 < line2:
            left_start = line1
            right_start = line2
        else:
            left_start = line2
            right_start = line1
        left_indexes = list()
        right_indexes = list()
        margin = int(self.search_box_size_margin)
        height = int(self.search_box_height)
        for start_x, lane_indexes in (left_start, left_indexes), (right_start, right_indexes):
            x_center = int(start_x)
            min_y = img.shape[0] - self.max_range
            for start_y in range(img.shape[0]-height, min_y, -height):
                end_y = start_y + height
                start_x = max(0, x_center - margin)
                end_x = min(x_center + margin, img.shape[1])
                roi = img[start_y:end_y, start_x:end_x]
                points = roi.nonzero()
                # Add the offsets back to the points
                points = (points[0] + start_y, points[1] + start_x)
                count = len(points[0])
                if count > self.recenter_thresh:
                    # Recenter the search box on the x-centroid of the lane
                    x_center = int(np.mean(points[1]))
                lane_indexes.append(points)
        return np.concatenate(left_indexes, 1), np.concatenate(right_indexes, 1)

    def find_peaks(self, data, order=None):
        '''
        Find the coordinates of the peaks in the 1-D vector, sorted by
        magnitude

        data -- Data to find peaks in
        '''
        if order is None:
            order = self.resolution/4
        peaks = find_peaks_cwt(data, np.arange(1, order))
        peaks = np.array(peaks)
        sort_order = np.argsort(data[peaks])
        sorted_peaks = np.flipud(peaks[sort_order])
        return sorted_peaks

    def smoothed_histogram(self, img):
        '''
        Calculate a histogram of the image over the x-axis, then smooth
        using a sliding window filter
        '''
        height = img.shape[0]
        histogram = np.sum(img, axis=0)
        window = int(self.resolution/4)
        w = np.ones(window, float)/window
        smoothed_histogram = np.convolve(histogram, w, 'same')
        return smoothed_histogram

    def find_two_lanes(self,
                       left_lane_points,
                       right_lane_points,
                       left_line,
                       right_line,
                       show_plot=False):
        '''
        Perform a joint polynomial fit that constrains the two lane lines
        to be parallel.

        left_lane_points -- list of coordinates of points in the left lane line
        right_lane_points -- list of coordinates of points in the right lane line
        '''
        x = np.concatenate((left_lane_points[1], right_lane_points[1]))
        n_left_points = len(left_lane_points[0])
        n_right_points = len(right_lane_points[0])
        y = np.concatenate((np.stack((left_lane_points[0]**2,
                                      left_lane_points[0],
                                      np.ones(n_left_points),
                                      np.zeros(n_left_points)), axis=1),
                            np.stack((right_lane_points[0]**2,
                                      right_lane_points[0],
                                      np.zeros(n_right_points),
                                      np.ones(n_right_points)), axis=1)))
        p = solve_least_squares(y, x)[0]
        if left_line is None:
            left_line = Line()
        if right_line is None:
            right_line = Line()
        left_line.setFit(np.array([p[0], p[1], p[2]]))
        right_line.setFit(np.array([p[0], p[1], p[3]]))

        if show_plot:
            plt.figure()
            plt.plot(left_lane_points[0], left_lane_points[1], 'r.')
            plt.plot(right_lane_points[0], right_lane_points[1], 'b.')
            plt.plot(left_lane_points[0], left_line.vals(left_lane_points[0]), 'r')
            plt.plot(right_lane_points[0], right_line.vals(right_lane_points[0]), 'b')

        return left_line, right_line

    def fit_lanes(self, img, last_left=None, last_right=None, show_plots=False):
        '''
        Find both lanes in the top-down binary lane image.

        img -- Binary top-down lane image to search
        last_left -- Previous detection of left lane line to use as a hint
            (optional)
        last_right -- Previous detection of the right lane line to use as a hint
            (optional)
        '''
        closed_img = self.close_img(img)
        # min_y is used to filter out furthest points
        min_y = img.shape[0] - self.max_range
        if last_left is None or last_right is None:
            left_lane_points, right_lane_points = self.find_lane_points(closed_img)
        else:
            y, x = img.nonzero()
            keep_y = y > min_y
            left_x_pred = last_left.vals(y)
            right_x_pred = last_right.vals(y)
            dx_left = np.abs(left_x_pred - x)
            dx_right = np.abs(right_x_pred - x)
            keep_left = (dx_left < self.search_box_size_margin) & keep_y
            keep_right = (dx_right < self.search_box_size_margin) & keep_y
            left_lane_points = (y[keep_left], x[keep_left])
            right_lane_points = (y[keep_right], x[keep_right])
        left_lane, right_lane = self.find_two_lanes(left_lane_points, right_lane_points,
                                                    last_left, last_right,
                                                    show_plots)
        if show_plots:
            plot_img = np.copy(closed_img)
            plot_img = cv2.cvtColor(plot_img*255, cv2.COLOR_GRAY2RGB)
            for pt in zip(left_lane_points[0], left_lane_points[1]):
                plot_img[pt] = (255, 0, 0)
            for pt in zip(right_lane_points[0], right_lane_points[1]):
                plot_img[pt] = (0, 0, 255)
            plt.figure()
            plt.imshow(plot_img)
            plt.plot(left_lane.vals(left_lane_points[0]), left_lane_points[0], 'g')
            plt.plot(right_lane.vals(right_lane_points[0]), right_lane_points[0], 'g')
        return left_lane, right_lane
