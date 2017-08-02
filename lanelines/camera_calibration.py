"""This module provides methods for calibrating the intrinsic parameters of a camera."""

import cv2
from matplotlib import pyplot as plt
import numpy as np
import sys


def draw_corners(img, board_size, corners):
    """Draw chessboard corners on an image, which can be color or grayscale."""
    if len(img.shape) < 3 or img.shape[2] < 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.drawChessboardCorners(img, board_size, corners, 1)


def calibrate_from_images(imgs, board_size, show_chessboards=False):
    """
    Calculate the camera matrix and distortion coefficients from images of a chessboard pattern.

    Arguments:
    imgs -- a list of (BGR) images.
    board_size -- the size of the chessboard to search for as a 2-element array-like,
        where the 0th element is the width and the 1st element is the height
    show_chessboards -- (Optional) if true, draw the found corners on the input images using
        pyplot's imshow method. This is useful for debug, but requires a window manager or
        Jupyter/IPython notebook.
    """
    img_pts = list()
    obj_pts = list()
    board = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    board[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    # Criteria for sub-pixel corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for name, img in imgs.items():
        # Load the image in grayscale mode
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(img, board_size, None)
        if ret is True:
            # Use subpixel corner finder to refine corner positions
            corners2 = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
            img_pts.append(corners2)
            obj_pts.append(board)
            if show_chessboards:
                plt.figure()
                plt.imshow(draw_corners(img, board_size, corners))
                plt.title(name)
        else:
            print("Could not find chessboard in image {}".format(name, ret), file=sys.stderr)
            if show_chessboards:
                plt.figure()
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
                plt.title(name)
    if show_chessboards:
        plt.show()
    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(np.array(obj_pts),
                                                  np.array(img_pts),
                                                  img.shape,
                                                  None,
                                                  None)
    return K, D
