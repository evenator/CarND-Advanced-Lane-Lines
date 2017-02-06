import cv2
from PIL import Image
from matplotlib import pyplot as plt

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

def Grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def XSobel(img, ksize=3):
    return cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)

def YSobel(img, ksize=3):
    return cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)

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
    def __init__(self, P):
        self._P = P
    def transformImage(self, src, output_size=None):
        '''
        Transform an image from camera perspective to top-down view
        '''
        if output_size is None:
            output_size = (src.shape[0], src.shape[1])
        return cv2.warpPerspective(src, self._P, output_size)
