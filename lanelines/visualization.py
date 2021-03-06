"""Various utility functions for drawing lanes and displaying images."""

import cv2
from matplotlib import pyplot as plt
import numpy as np


def comparison_plot(img1, img2, label1, label2, top_label):
    """
    Plot two images side-by-side for comparison as a PyPlot figure.

    img1 -- The (BGR) image to show in the left subplot
    img2 -- The (BGR) image to show in the right subplot
    label1 -- The title of the left subplot
    label2 -- The title of the right subplot
    top_label -- The overall title of the figure

    Returns The PyPlot figure handle
    """
    f, (left, right) = plt.subplots(1, 2)
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    left.imshow(img1)
    left.set_title(label1)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    right.imshow(img2)
    right.set_title(label2)
    f.suptitle(top_label)
    return f


def draw_lane(lane, img_shape, resolution):
    """
    Draw the filled-in lane on the ground image.

    lane -- Lane object representing the left lane
    img_shape -- Shape of the output image **in pixels** as (width, height) tuple
    resolution -- Resolution of the output image in pixels/meter
    """
    canvas = np.zeros((img_shape[0], img_shape[1], 3), np.uint8)
    y_pixels = range(0, img_shape[0], int(resolution/10))  # A polyline with points every 10 cm
    pts_left = lane.left.vals(y_pixels).astype(int)
    pts_right = lane.right.vals(y_pixels).astype(int)
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([pts_left, y_pixels]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([pts_right, y_pixels])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    green = (0, 255, 0)
    cv2.fillPoly(canvas, np.int_([pts]), green)
    return canvas


def plot_on_img(img, *lines, color='b'):
    """
    Plot an arbitrary number of arbitrary Lines on an image.

    Note that the independent variable is **y**, not x, and that image coordinate conventions apply.

    img -- Image (BGR) to plot on top of
    lines -- Any number of Line types
    color -- Matplotlib color (default 'b') to plot the polynomials in

    Returns a 3-channel numpy image
    """
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    fig = plt.figure()
    if len(img.shape) < 3 or img.shape[2] < 3:
        plt.imshow(img, cmap='gray')
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
    for line in lines:
        plotx = line.vals(poly, ploty)
        plt.plot(plotx, ploty, color=color)
    fig.canvas.draw()
    out_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    out_img = out_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return out_img
