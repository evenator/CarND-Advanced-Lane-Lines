from datatypes import Line

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
    def __init__(self, P, output_size=None):
        '''
        Create a GroundProjector from a Perspective matrix

        P -- Perspective transform matrix, as created by
            cv2.getPerspectiveTransform
        output_size -- Default shape of the output image **in pixels** as a
            tuple (width, height). If set to None, there is no default output
            shape.
        '''
        self._P = P
        self._output_size = output_size

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

    def inverseTransformImage(self, src, output_shape):
        '''
        Transform an image from top-down view to camera perspective

        src -- Image to transform, as a numpy Array-like
        output_shape -- Shape of the output image **in pixels** as a tuple
            (width, height).
        '''
        return cv2.warpPerspective(src, self._P, output_shape[1::-1], flags=cv2.WARP_INVERSE_MAP)

    @classmethod
    def from_point_correspondence(cls, image_pts, object_points, output_resolution, output_size=None):
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
    def __init__(self, sobel_kernel_size, direction_thresh, gradient_mag_thresh):
        '''
        Constructor

        sobel_kernel_size -- Size of kernel for gradient calculate (odd number
            of pixels)
        direction_thresh -- (min, max) tuple of thresholds for the direction
            angle of the gradient. Angles are in the range [0, pi/2]
        gradient_mag_thresh -- (min, max) tuple of thresholds for the magnitude
            of the gradient
        '''
        self._k = sobel_kernel_size
        self._direction_thresh = direction_thresh
        self._gradient_mag_thresh = gradient_mag_thresh

    def extract_lanes(self, img, show_plots=False):
        '''
        Process an RGB image and return a binary image

        img -- The RGB image to process (in camera perspective)
        show_plots -- Show intermediate images in PyPlot figures (default False)

        Returns a binary image in the camera perspective
        '''
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]
        sobel_x = XSobel(s_channel, ksize=self._k)
        sobel_y = YSobel(s_channel, ksize=self._k)
        dxy = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        magnitude_scaled = scaled_abs(dxy)
        magnitude_binary = np.logical_and(magnitude_scaled >= self._gradient_mag_thresh[0],
                                          magnitude_scaled <= self._gradient_mag_thresh[1])
        if show_plots:
            plt.figure()
            plt.title('Magnitude Binary')
            plt.imshow(magnitude_binary, cmap='gray')
        direction = np.arctan2(np.abs(sobel_y), np.abs(sobel_x))
        direction_binary = np.logical_and(direction >= self._direction_thresh[0],
                                          direction <= self._direction_thresh[1])
        if show_plots:
            plt.figure()
            plt.title('Direction Binary')
            plt.imshow(direction_binary, cmap='gray')
        lanes = np.dstack([magnitude_binary, direction_binary]).all(-1)
        return np.uint8(lanes)

# TODO: Restructure this class and start using hints
# TODO: Rename private methods
# TODO: Add option to plot intermedia images
class LaneFitter(object):
    '''
    A Processor that takes in a binary lane image in top-down perspective and
    finds the lane lines.
    '''
    def __init__(self, resolution):
        '''
        Constructor

        resolution -- Resolution of the input image in meters/pixel
        '''
        self.resolution = resolution
        self.search_box_size_margin = resolution
        self.search_box_height = int(resolution/2)
        self.recenter_thresh = 1000

    def close_img(self, img):
        '''
        Perform a morphological closing on the image to join clusters of
        separate pixels in blobs.
        '''
        kernel_size = int(self.resolution/2)
        morph_kernel = np.ones((int(self.resolution/2), int(self.resolution/4)))
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, morph_kernel)

    def find_lane(self, img, start_x, hint = None):
        '''
        Find the lane line that begins at the position `start_x` at the bottom
        of the image and fit a quadratic polynomial to it. Uses the `hint` line
        to mask off likely pixels. If no hint` is given, performs a search using
        `find_lane_points`.

        img -- Binary lane image in top-down perspective to search
        start_x -- The rough position of the lane line's beginning at the bottom
            of the image
        hint -- A lane line to use as a hint to avoid having to do a sliding
            window search, i.e. a detection in a previous frame (optional)
        '''
        lane_indexes = self.find_lane_points(img, start_x, hint)
        line = Line()
        line.closest_y = img.shape[0]
        line.poly = np.polyfit(lane_indexes[0], lane_indexes[1], 2)
        return line

    def find_lane_points(self, img, start_x):
        '''
        Find a single lane using a sliding filter. Initialize the
        sliding filter at the bottom of the image using start_x.

        img -- Image to search in
        start_x -- X coordinate to start the search
        '''
        nonzero = img.nonzero()
        nonzero = zip(nonzero[0], nonzero[1])
        margin = int(self.search_box_size_margin)
        height = int(self.search_box_height)
        x_center = int(start_x)
        lane_indexes = list()
        for start_y in range(img.shape[0]-height, 0, -height):
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
        lane_indexes = np.concatenate(lane_indexes, 1)
        return lane_indexes

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

    def find_two_lanes(self, left_lane_points, right_lane_points):
        '''
        Perform a joint polynomial fit that constrains the two lane lines
        to be parallel.

        left_lane_points -- list of coordinates of points in the left lane line
        right_lane_points -- list of coordinates of points in the right lane line
        '''
        x = np.concatenate((left_lane_points[1], right_lane_points[1]))
        print("X: {}".format(x.shape))
        n_left_points = len(left_lane_points[0])
        n_right_points = len(right_lane_points[0])
        y = np.concatenate((np.stack((left_lane_points[0]**2, left_lane_points[0], np.ones(n_left_points), np.zeros(n_left_points)), axis=1),
                            np.stack((right_lane_points[0]**2, right_lane_points[0], np.zeros(n_right_points), np.ones(n_right_points)), axis=1)))
        print("Y: {}".format(y.shape))
        p = solve_least_squares(y, x)[0]
        print("p: {}".format(p))
        left_line = Line()
        right_line = Line()
        left_line.poly = np.array([p[0], p[1], p[2]])
        right_line.poly = np.array([p[0], p[1], p[3]])

        plt.figure()
        plt.plot(left_lane_points[0], left_lane_points[1], 'b.')
        plt.plot(right_lane_points[0], right_lane_points[1], 'r.')
        plt.plot(left_lane_points[0], left_line.vals(left_lane_points[0]))
        plt.plot(right_lane_points[0], right_line.vals(right_lane_points[0]))
        plt.show()

        return left_line, right_line

    def fit_lanes(self, img, last_left = None, last_right = None):
        '''
        Find both lanes in the top-down binary lane image.

        img -- Binary top-down lane image to search
        last_left -- Previous detection of left lane line to use as a hint
            (optional)
        last_right -- Previous detection of the right lane line to use as a hint
            (optional)
        '''
        height = img.shape[0]
        width = img.shape[1]
        closed_img = self.close_img(img)
        smoothed_histogram = self.smoothed_histogram(closed_img[int(height/2):,:])
        sorted_peaks = self.find_peaks(smoothed_histogram)
        if sorted_peaks[0] < sorted_peaks[1]:
            left_start = sorted_peaks[0]
            right_start = sorted_peaks[1]
        else:
            left_start = sorted_peaks[1]
            right_start = sorted_peaks[0]
        left_lane_points = self.find_lane_points(img, left_start)
        right_lane_points = self.find_lane_points(img, right_start)
        left_lane, right_lane = self.find_two_lanes(left_lane_points, right_lane_points)
        return left_lane, right_lane
