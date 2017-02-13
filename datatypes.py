import numpy as np

class Line(object):
    '''
    An object to represent a detected lane line
    '''
    def __init__(self):
        self.detected = False
        self.poly = [] # Polynomial coefficients
        self.closest_y = 0
        self.resolution = 200 # pixels/meter
    def radius(self):
        """ Return the radius of curvature in meters"""
        A = self.poly[0]
        B = self.poly[1]
        y = self.closest_y
        r_pix = (1.0 + (2.0 * A * y  + B)**2)**(1.5) / abs(2 * A)
        return r_pix / self.resolution
    def curvature(self):
        """ Return the curvature in 1/meters"""
        return 1.0/self.radius()
    def vals_m(self, y_vals):
        '''
        Calculate x values (meters) for the given series of y_vals (meters)
        '''
        y_vals = np.array(y_vals) * self.resolution
        return self.vals(y_vals) / self.resolution
    def vals(self, y_vals):
        '''
        Calculate x values (pixels) for the given series of y_vals (pixels)
        '''
        return np.polyval(self.poly, y_vals)
