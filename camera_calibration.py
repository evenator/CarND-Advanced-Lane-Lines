from processors import Undistorter
from util import comparison_plot

import argparse
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import sys

def Grayscale(img):
    '''Convert an image to Grayscale'''
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def cornersImg(img, board_size, corners):
    '''
    Draw chessboard corners on an image, which can be color or grayscale.
    '''
    if len(img.shape) < 3 or img.shape[2] < 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.drawChessboardCorners(img, board_size, corners, 1)


def calibrateFromImgs(imgs, board_size, show_chessboards=False):
    '''
    Calculate the camera matrix and distortion coefficients from a set of
    images.
    '''
    img_pts = list()
    obj_pts = list()
    board = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    board[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    # Criteria for sub-pixel corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for name, img in imgs.items():
        # Load the image in grayscale mode
        img = Grayscale(img)
        ret, corners = cv2.findChessboardCorners(img, board_size, None)
        if ret is True:
            # Use subpixel corner finder to refine corner positions
            corners2 = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
            img_pts.append(corners2)
            obj_pts.append(board)
            if show_chessboards:
                plt.figure()
                plt.imshow(cornersImg(img, board_size, corners))
                plt.title(name)
        else:
            print("Could not find chessboard in image {}".format(name, ret), file=sys.stderr)
            if show_chessboards:
                plt.figure()
                plt.imshow(img)
                plt.title(name)
    if show_chessboards:
        plt.show()
    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(np.array(obj_pts), np.array(img_pts), img.shape, None, None)
    return K, D


def main():
    parser = argparse.ArgumentParser(description='Calculate and save the camera matrix and distortion coefficients from a directory of chessboard images')
    parser.add_argument('--board-size', nargs=2, type=int, default=(9, 6), help='Two-tuple containing the width and height of the chessboard (number of corners)')
    parser.add_argument('--show-chessboards', action='store_true', help='Display chessboards with corners found')
    parser.add_argument('--show-undistorted', action='store_true', help='Show images after distortion-correction to verify calibration')
    parser.add_argument('directory', type=str)
    args = parser.parse_args()
    imgs = dict()
    for img_file in os.listdir(args.directory):
        # Load the image in grayscale mode
        imgs[img_file] = cv2.imread(os.path.join(args.directory, img_file))
    K, D = calibrateFromImgs(imgs, args.board_size, args.show_chessboards)
    print("Camera Matrix:")
    print(K)
    print("Distortion Coeffeficients:")
    print(D)

    print("Saving camera matrix to camera_matrix.npy")
    np.save('camera_matrix.npy', K)
    print("Saving distortion coefficients to disortion_coefficients.npy")
    np.save('disortion_coefficients.npy', D)
    undistorter = Undistorter(K, D)

    if args.show_undistorted:
        for fname, img in imgs.items():
            # Load the image in grayscale mode
            undistorted_img = undistorter.undistortImage(img)
            undistorted_img = undistorted_img[:, :, ::-1] # RGB<->BGR flip
            img = img[:, :, ::-1] # RGB<->BGR flip
            comparison_plot(img, undistorted_img, 'Raw', 'Undistorted', fname)
        plt.show()


if __name__ == "__main__":
    main()
