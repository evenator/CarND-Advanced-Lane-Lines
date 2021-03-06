"""Datatypes to represent lane geometry."""

import numpy as np


class Lane(object):
    """A lane is composed of a left and right Line."""

    def __init__(self, left, right):
        """
        Constructor for a Lane.

        Arguments:
        left -- A Line representing the left lane line.
        right -- A Line representing the right lane line.
        """
        self.left = left
        self.right = right

    def curvature(self):
        """
        Calculate and return the curvature of the lane.

        The curvature is measured at the centerline in units of 1/meter.
        """
        return 2.0 / (self.left.radius() + self.right.radius())

    def camera_position(self):
        """
        Calculate and return the camera position relative to the center of the Lane.

        Calculate and return the lateral offset of the camera from the center of the lane in
        meters. Positive values indicate the camera is left of the center line, and negative values
        indicate the camera is to the right of the center line.
        """
        return (self.left.dist_from_center_m() + self.right.dist_from_center_m()) / 2.0

    def valid(self):
        """Check whether the lane meets validity criteria."""
        # Check that both lines detected
        if self.left is None or self.right is None:
            print("Both lines not detected")
            return False
        # Check vehicle is in lane
        veh_pos = (self.right.dist_from_center_m() + self.left.dist_from_center_m())/2
        if abs(veh_pos) > 1.0:
            print("Vehicle is not in the center of the lane (position={})".format(veh_pos))
            return False
        # Check lines parallel
        # by checking the variance of the widths at many points
        y_vals = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]
        widths = self.left.vals_m(y_vals) - self.right.vals_m(y_vals)
        width_var = np.var(widths)
        if width_var > 0.2:
            print("Lines not parallel (width variance={})".format(width_var))
            return False
        # Check curvature is sane
        # See http://onlinemanuals.txdot.gov/txdotmanuals/rdw/horizontal_alignment.htm
        mean_curvature = 2.0/(self.left.radius() + self.right.radius())
        if mean_curvature > 0.005679:  # Curvature in 1/m for radius = 587 ft
            print("Curvature is too large (curvature={})".format(mean_curvature))
            return False
        return True


class ExponentialFilter(object):
    """
    Exponential smoothing filter.

    Implements a filter defined by the update equation:
    s = alpha * x + (1 - alpha) * s
    """

    def __init__(self, alpha, init=0.0):
        """
        Construct an Exponential filter with smoothing value alpha.

        Arguments:
        alpha -- Exponential smoothing factor
        init -- (Optional) Value to initialize the filter to. The default value is 0.0.
        """
        self._alpha = alpha
        self.s = init

    def __call__(self, x):
        """Update the filter with the new measurment x and return the filtered value."""
        self.s = self._alpha * x + (1 - self._alpha) * self.s
        return self.s


class Line(object):
    """
    An object to represent a detected lane line.

    The polynomial fit is updated with an exponential filter, which reduces jitter when the line
    is used for estimation in a video pipeline.
    """

    def __init__(self):
        """
        Construct a Line with default values.

        The Line is not initialized until update_fit as been called at least once.
        """
        self.poly = []         # Polynomial coefficients
        self.closest_y = 0     # Pixel y-coordinate closest to vehicle (img bottom)
        self.middle_x = 0      # Pixel x-coordinate clostest to vehicle (img center)
        self.resolution = 200  # Image pixels/meter
        self.age = 0           # Number of frames since last matched

    def radius(self):
        """Return the radius of curvature in meters."""
        A = self.poly[0]
        B = self.poly[1]
        y = self.closest_y
        r_pix = (1.0 + (2.0 * A * y + B)**2)**(1.5) / abs(2 * A)
        return r_pix / self.resolution

    def dist_from_center_m(self):
        """Return the distance in meters of the bottom of the line from the center of the image."""
        x = self.vals([self.closest_y])[0]
        return (x - self.middle_x) / self.resolution

    def curvature(self):
        """Return the curvature in 1/meters."""
        return 1.0/self.radius()

    def vals_m(self, y_vals):
        """
        Calculate x values (meters) for the given series of y_vals (meters).

        y_vals -- Series of y values (meters) in distance from the vehicle
            (0 is closest to the vehicle)
        """
        y_vals = np.array(y_vals)
        y_vals = (np.ones_like(y_vals) * self.closest_y - y_vals) * self.resolution
        return self.vals(y_vals) / self.resolution

    def vals(self, y_vals):
        """Calculate x values in pixels for the given series of y_vals in pixels."""
        return np.polyval(self.poly, y_vals)

    def update_fit(self, coefficients):
        """Update the line fit with a new detection."""
        if not hasattr(self, 'filters'):
            self.filters = [ExponentialFilter(alpha=0.2, init=x) for x in coefficients]
        self.poly = [f(x) for x, f in zip(coefficients, self.filters)]
