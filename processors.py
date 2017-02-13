from datatypes import Line
from util import XSobel, YSobel, scaled_abs

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks_cwt

class Undistorter(object):
    def __init__(self, K, D):
        self._K = K
        self._D = D

    def undistortImage(self, src):
        '''
        Undistort the raw camera image
        '''
        return cv2.undistort(src, self._K, self._D)

class GroundProjector(object):
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


class LaneExtractor(object):
    def __init__(self, sobel_kernel_size, direction_thresh, gradient_mag_thresh):
        self._k = sobel_kernel_size
        self._direction_thresh = direction_thresh
        self._gradient_mag_thresh = gradient_mag_thresh

    def extract_lanes(self, img, show_plots=False):
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

class LaneFitter(object):
    def __init__(self, resolution):
        self.resolution = resolution
        self.search_box_size_margin = resolution
        self.search_box_height = int(resolution/2)
        self.recenter_thresh = 1000
    
    def close_img(self, img):
        kernel_size = int(self.resolution/2)
        morph_kernel = np.ones((int(self.resolution/2), int(self.resolution/4)))
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, morph_kernel)
    
    def find_lane(self, img, start_x, hint = None):
        # TODO: Use hints
        lane_indexes = self.find_lane_points(img, start_x, hint)
        line = Line()
        line.closest_y = img.shape[0]
        line.poly = np.polyfit(lane_indexes[0], lane_indexes[1], 2)
        return line
    
    def find_lane_points(self, img, start_x, hint=None):
        '''
        Find a single lane using a sliding filter. Initialize the
        sliding filter at the bottom of the image using start_x.
        img -- Image to search in
        start_x -- X coordinate to start the search
        hint -- A line object to use as a hint (may be the last detection)
                Currently unused.
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
        Find the coordinates of the peaks in the 1-D vector, sorted by height
        
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
        
    def fit_lanes(self, img, last_left = None, last_right = None):
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
        # TODO: Joint-fit
        left_lane = self.find_lane(img, left_start)
        right_lane = self.find_lane(img, right_start)
        return left_lane, right_lane
