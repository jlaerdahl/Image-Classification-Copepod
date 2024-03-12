from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng
rng.seed(12345)

def thresh_callback(val):

    threshold = val
    
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    
    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # Find the rotated rectangles and ellipses for each contour
    minEllipse = [None]*len(contours)
    for i, c in enumerate(contours):
        if c.shape[0] > 10:
            minEllipse[i] = cv.fitEllipse(c)
    # Draw contours + rotated rects + ellipses
    
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    
    for i, c in enumerate(contours):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        # ellipse
        if c.shape[0] > 10:
            # contour
            cv.drawContours(drawing, contours, i, color)
            cv.ellipse(drawing, minEllipse[i], color, 2)
    
    cv.imshow('Contours', drawing)
    cv.imshow('Canny', canny_output)
    # cv.imshow("Img", src)
    # cv.imshow("Intersection", intersection)
    
# parser = argparse.ArgumentParser(description='Code for Creating Bounding rotated boxes and ellipses for contours tutorial.')
# parser.add_argument('--input', help='Path to input image.', default='nauplii.jpg')
# args = parser.parse_args()
# src = cv.imread(cv.samples.findFile(args.input))
# if src is None:
#     print('Could not open or find the image:', args.input)
#     exit(0)
# # Convert image to gray and blur it

image_file = "copepod.jpg"

# Read the image
image = cv.imread(image_file)

src_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
source_window = 'Image'
cv.namedWindow(source_window)
cv.imshow(source_window, src_gray)
max_thresh = 255
thresh = 100 # initial threshold
cv.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
thresh_callback(thresh)
cv.waitKey()
