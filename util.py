import cv2
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

def save_animation(filename, imgs, duration):
    frame1 = Image(imgs[0])
    frame1.save(filename, save_all=True, duration=duration, loop=-1, append_images=[Image(i) for i in imgs[1:]])

def comparison_plot(img1, img2, label1, label2, top_label):
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
    ''' Plot an arbitrary number of arbitrary polynomials on an image'''
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

def Grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def XSobel(img, ksize=3):
    return cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)


def scaled_abs(img):
    abs_img = np.abs(img)
    return np.uint8(255 * abs_img / np.max(abs_img))

def YSobel(img, ksize=3):
    return cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
