"""
Created on Tue Jun 02 21:31:17 2015
Canny Edge Detection tool with trackbars for varying thresholds.
@author: Johnny
"""

import cv2


# this function is needed for the createTrackbar step downstream
def nothing(x):
    pass

path = "assets/car.jpg"
# read the experimental image
img = cv2.imread(path, 0)

# create trackbar for canny edge detection threshold changes
cv2.namedWindow('canny')

# add ON/OFF switch to "canny"
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'canny', 0, 1, nothing)

# add lower and upper threshold slidebars to "canny"
cv2.createTrackbar('lower', 'canny', 0, 500, nothing)
cv2.createTrackbar('upper', 'canny', 0, 500, nothing)

# Infinite loop until we hit the escape key on keyboard
while(1):

    # get current positions of four trackbars
    lower = cv2.getTrackbarPos('lower', 'canny')
    upper = cv2.getTrackbarPos('upper', 'canny')
    s = cv2.getTrackbarPos(switch, 'canny')

    if s == 0:
        edges = img
    else:
        edges = cv2.Canny(img, lower, upper)

    edges = cv2.GaussianBlur(edges, (17,17), 0)
    edges2 = cv2.Canny(edges, lower, upper)

    # display images
    cv2.imshow('original', img)
    cv2.imshow('canny', edges)
    cv2.imshow('canny2', edges2)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:   # hit escape to quit
        break

cv2.destroyAllWindows()

