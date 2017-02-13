import cv2
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

'''
Various utility functions for drawing lanes and displaying images
'''

def save_animation(filename, imgs, duration):
    '''
    Create a looping animated GIF image

    filename -- Path to save the image
    imgs -- Array-like of frames for the animation (must all be the same shape)
    duration -- Duration (seconds) to show each frame
    '''
    frame1 = Image(imgs[0])
    frame1.save(filename, save_all=True, duration=duration, loop=-1, append_images=[Image(i) for i in imgs[1:]])

def comparison_plot(img1, img2, label1, label2, top_label):
    '''
    Plot two images side-by-side for comparison as a PyPlot figure

    img1 -- The image to show in the left subplot
    img2 -- The image to show in the right subplot
    label1 -- The title of the left subplot
    label2 -- The title of the right subplot
    top_label -- The overall title of the figure

    Returns The PyPlot figure handle
    '''
    f, (left, right) = plt.subplots(1, 2)
    left.imshow(img1)
    left.set_title(label1)
    right.imshow(img2)
    right.set_title(label2)
    f.suptitle(top_label)
    return f

def draw_lane(left_line, right_line, img_shape, resolution):
    '''
    Draw the filled-in lane on the ground image

    left_lane -- Line object representing the left lane
    right_lane -- Line object representing the right lane
    img_shape -- Shape of the output image **in pixels** as (width, height) tuple
    resolution -- Resolution of the output image in pixels/meter
    '''
    canvas = np.zeros((img_shape[0], img_shape[1], 3), np.uint8)
    y_pixels = range(0, img_shape[0], int(resolution/10)) # A polyline with points every 10 cm
    pts_left = left_line.vals(y_pixels).astype(int)
    pts_right = right_line.vals(y_pixels).astype(int)
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([pts_left, y_pixels]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([pts_right, y_pixels])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(canvas, np.int_([pts]), (0,255, 0))
    return canvas

def plot_on_img(img, *poly, color='b'):
    '''
    Plot an arbitrary number of arbitrary polynomials on an image. Note that
    the independent variable is **y**, not x, and that image coordinate
    conventions apply

    img -- Image to plot on top of
    poly -- Any number of polynomials in pixel space, expressed as numpy-style
        coefficient vectors
    color -- Matplotlib color (default 'b') to plot the polynomials in

    Returns a 3-channel numpy image
    '''
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    fig = plt.figure()
    if len(img.shape) < 3 or img.shape[2] < 3:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    for poly in poly:
        plotx = np.polyval(poly, ploty)
        plt.plot(plotx, ploty, color=color)
    fig.canvas.draw()
    out_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    out_img = out_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return out_img
