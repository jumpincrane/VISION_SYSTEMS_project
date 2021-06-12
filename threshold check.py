import numpy as np
import cv2
import os

def nothing(x):
    pass


def last():
    bgr_img = cv2.imread('./images/img_000.jpg')
    bgr_img = cv2.resize(bgr_img, (int(bgr_img.shape[1] / 5), int(bgr_img.shape[0] / 5)))
    bgr_img = cv2.medianBlur(bgr_img, 5)
    laplace = cv2.Laplacian(bgr_img, cv2.CV_8U)
    cv2.normalize(laplace, bgr_img, 0, 600, cv2.NORM_MINMAX)
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    ret, gray_img = cv2.threshold(gray_img, 15, 255, cv2.THRESH_BINARY)


# gr_img = 255 -gr_im
    cv2.namedWindow('image')
    cv2.createTrackbar('THMIN', 'image', 0, 255, nothing)
    cv2.createTrackbar('THMAX', 'image', 0, 255, nothing)
    cv2.setTrackbarPos('THMIN', 'image', 12)
    cv2.setTrackbarPos('THMAX', 'image', 255)

    thmin = 0
    thmax = 50

    while 1:
        thmin = cv2.getTrackbarPos('THMIN', 'image')
        thmax = cv2.getTrackbarPos('THMAX', 'image')

        edges = cv2.Canny(gray_img, thmin, thmax)
        cv2.imshow('image', edges)
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break

last()