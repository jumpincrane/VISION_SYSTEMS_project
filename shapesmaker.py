import cv2
import numpy as np
import matplotlib.pyplot as plt


def normalize_filled(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = 255 - img
    ret, thresh = cv2.threshold(img, 125, 255, cv2.THRESH_OTSU)

    cnt, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    fil_cnts = []
    for c in cnt:
        area = cv2.contourArea(c)
        if area > 150:
            fil_cnts.append(c)
    # fill shape
    cv2.fillPoly(thresh, pts=fil_cnts, color=(255, 255, 255))

    bounding_rect = cv2.boundingRect(fil_cnts[0])
    img_cropped_bounding_rect = thresh[bounding_rect[1]:bounding_rect[1] + bounding_rect[3],
                                bounding_rect[0]:bounding_rect[0] + bounding_rect[2]]
    # resize all to same size
    img_resized = cv2.resize(img_cropped_bounding_rect, (600, 600))
    return img_resized


imgcircle = cv2.imread('./training/7/img_003(10).jpg')
imgworm = cv2.imread('./training/14/img_002(36).jpg')
imgbear = cv2.imread('./training/1/img_001(14).jpg')

imgs = [imgcircle, imgworm, imgbear]
imgs = [normalize_filled(i) for i in imgs]

cv2.imwrite('./circle.jpg', imgs[0])
cv2.imwrite('./worm.jpg', imgs[1])
cv2.imwrite('./bear.jpg', imgs[2])

for i in range(1, 4):
    plt.subplot(2, 3, i), plt.imshow(imgs[i - 1], cmap='gray')
    print(cv2.matchShapes(imgs[0], imgs[i - 1], 1, 0.0))
plt.show()
