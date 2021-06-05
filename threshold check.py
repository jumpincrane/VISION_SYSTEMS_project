import numpy as np
import cv2
import os

def nothing(x):
    pass

bgr_img = cv2.imread('./images/img_000.jpg')
bgr_img = cv2.resize(bgr_img, (int(bgr_img.shape[1] / 5), int(bgr_img.shape[0] / 5)))
# cv2.imshow("org", bgr_img)
bgr_img = cv2.medianBlur(bgr_img, 3)
# cv2.imshow("blur", bgr_img)
bgr_img = cv2.convertScaleAbs(bgr_img, alpha=1.2, beta=1)
# cv2.imshow("contrast", bgr_img)
hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

green_low = np.array([33, 30, 40])
green_high = np.array([75, 255, 255])
curr_mask = cv2.inRange(hsv_img, green_low, green_high)
hsv_img[curr_mask > 0] = ([0, 0, 0])
bgr_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
gr_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
cv2.namedWindow('image')

gr_img = 255 -gr_img
cv2.createTrackbar('THMIN', 'image', 0, 255, nothing)
cv2.createTrackbar('THMAX', 'image', 0, 255, nothing)
cv2.setTrackbarPos('THMIN', 'image', 0)
cv2.setTrackbarPos('THMAX', 'image', 50)

thmin = 0
thmax = 50

while(1):
    thmin = cv2.getTrackbarPos('THMIN', 'image')
    thmax = cv2.getTrackbarPos('THMAX', 'image')

    ret, threshold = cv2.threshold(gr_img, thmin, thmax, cv2.THRESH_BINARY)

    cv2.imshow('image', threshold)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break