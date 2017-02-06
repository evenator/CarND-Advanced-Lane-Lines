import cv2

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
