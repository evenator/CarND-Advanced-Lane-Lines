import numpy as np

class Line(object):
    '''
    An object to represent a detected lane line
    '''
    def __init__(self):
        self.poly = []         # Polynomial coefficients
        self.closest_y = 0     # Pixel y-coordinate closest to vehicle (img bottom)
        self.middle_x = 0      # Pixel x-coordinate clostest to vehicle (img center)
        self.resolution = 200  # Image pixels/meter
    def radius(self):
        """ Return the radius of curvature in meters"""
        A = self.poly[0]
        B = self.poly[1]
        y = self.closest_y
        r_pix = (1.0 + (2.0 * A * y  + B)**2)**(1.5) / abs(2 * A)
        return r_pix / self.resolution
    def dist_from_center_m(self):
        x = self.vals([self.closest_y])[0]
        return (x - self.middle_x) / self.resolution
    def curvature(self):
        """ Return the curvature in 1/meters"""
        return 1.0/self.radius()
    def vals_m(self, y_vals):
        '''
        Calculate x values (meters) for the given series of y_vals (meters)

        y_vals -- Series of y values (meters) in distance from the vehicle
            (0 is closest to the vehicle)
        '''
        y_vals = np.array(y_vals)
        y_vals = (np.ones_likes(y_vals) * self.closest_y - y_vals) * self.resolution
        return self.vals(y_vals) / self.resolution
    def vals(self, y_vals):
        '''
        Calculate x values (pixels) for the given series of y_vals (pixels)
        '''
        return np.polyval(self.poly, y_vals)
    def setFit(self, coefficients):
        '''
        Set the current fit to a list of polynomial coefficients
        '''
        self.poly = coefficients
