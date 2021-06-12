import numpy as np
import cv2
import os
import sys
import json
from math import *


class DetectApp:
    def __init__(self):
        self.images = []
        self.images_path = './images'
        self.result_path = './'
        self.result = {}
        self.image_names = []
        self.objects = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.circle_temp = cv2.imread('./config/circle.jpg', cv2.IMREAD_GRAYSCALE)
        self.worm_temp = cv2.imread('./config/worm.jpg', cv2.IMREAD_GRAYSCALE)
        self.bear_temp = cv2.imread('./config/bear.jpg', cv2.IMREAD_GRAYSCALE)
        self.gt_images = [(4, 5, 3), (10, 10, 6), (15, 15, 9), (7, 3, 2), (7, 22, 2), (7, 22, 9), (7, 22, 9),
                          (11, 24, 0), (16, 24, 1), (17, 36, 9), (8, 10, 5), (13, 16, 10), (13, 16, 10),
                          (14, 16, 12), (16, 14, 9)]

    def upload_images(self):
        for root, dirs, files in os.walk(self.images_path):
            for f in files:
                if f.endswith('.jpg'):
                    img_path = os.path.join(root, f)
                    img = cv2.imread(img_path)
                    self.images.append(img)
                    self.image_names.append(f)

    def process(self):
        for i in range(len(self.images)):
            print(f"Image_{i}")
            img = self.images[i]

            bgr_img = self.mask_color_adjust_color(img)
            h, s, v = cv2.split(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV))
            # make threshold img
            # dilate = cv2.dilate(s, (3, 3))
            # closing = cv2.morphologyEx(s, cv2.MORPH_CLOSE, (5, 5), iterations=4)

            ret, threshold = cv2.threshold(s, 0.43, 1.0, cv2.THRESH_BINARY)
            threshold = np.uint8(threshold)

            countours, hierarchy = cv2.findContours(threshold, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

            bears = []
            worms = []
            circles = []
            for j, cnt in enumerate(countours):
                area = cv2.contourArea(cnt)
                if area > 400:
                    # mask found object
                    mask_object = np.zeros(bgr_img.shape[:2], dtype='uint8')
                    cv2.fillPoly(mask_object, pts=[cnt], color=255)
                    # find center of mass with moments
                    M = cv2.moments(mask_object)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    # find bounding rectangle to the contour and angle
                    rect = cv2.minAreaRect(cnt)
                    w = rect[1][0]
                    h = rect[1][1]
                    box = cv2.boxPoints(rect)
                    # calculate parameters of cnt
                    cnt_area = area
                    cnt_center = (cx, cy)
                    cnt_perimeter = cv2.arcLength(cnt, True)  # obwod
                    cnt_eccentricity = self.get_eccentricity(M)
                    cnt_solidity = self.get_solidity(cnt_area, cnt)
                    cnt_compactness = cnt_area / (w * h)

                    print(f'area: {cnt_area}, center: {cnt_center}, obwod: {cnt_perimeter},'
                          f' eccentricity: {cnt_eccentricity}, solidity: {cnt_solidity},'
                          f' prostokatnosc: {cnt_compactness}')

                    # calculate ratio of areas ~ 2 = circle
                    r_w = w / 2
                    r_h = h / 2
                    circle_area_w = np.pi * r_w ** 2
                    circle_area_h = np.pi * r_h ** 2
                    extent = (circle_area_h + circle_area_w) / area
                    print(extent)
                    if 1.7 < extent < 2.04:
                        circles.append((cx, cy, mask_object))
                        print('circles')
                    elif 2.04 < extent < 3.7:
                        bears.append((cx, cy, mask_object))
                        print('bears')
                    elif extent > 3.7:
                        worms.append((cx, cy, mask_object))
                        print('worm')

                    # draw bounding box and contours
                    box = np.int0(box)
                    cv2.drawContours(bgr_img, [box], -1, (0, 0, 255), 2)
                    cv2.drawContours(bgr_img, cnt, -1, (0, 255, 0), 3)
                    cv2.imshow('bgr', bgr_img)
                    cv2.waitKey()

            # ======= results ============
            # cv2.imshow('thresh', threshold)
            print('Bears_succs: ', len(bears), self.gt_images[i][1])
            print('circles_succs: ', len(circles), self.gt_images[i][0])
            print('worms_succs: ', len(worms), self.gt_images[i][2])

            cv2.imshow('bgr', bgr_img)
            cv2.waitKey()

        # with open(self.result_path, 'w') as outfile:
        #     json.dump(self.result, outfile)

    @staticmethod
    def mask_color_adjust_color(image):
        img = image / np.float32(255)
        img **= 2.4
        img = cv2.pyrDown(cv2.pyrDown(img))

        gray_w = False
        if gray_w:
            gray = np.mean(img, axis=(0, 1))
        else:
            gray = np.median(img, axis=(0, 1))

        correction = gray.mean() / gray
        img *= correction
        img **= (1 / 2.4)

        return img

    @staticmethod
    def get_eccentricity(moment):
        bigSqrt = sqrt((moment['mu20'] - moment['mu02']) * (moment['mu20'] - moment['mu02']) + 4 * moment['mu11'] *
                       moment['mu11'])
        ecc = (moment['mu20'] + moment['mu02'] - bigSqrt) / (moment['mu20'] + moment['mu02'] + bigSqrt)

        return ecc

    @staticmethod
    def get_solidity(area, cnt):
        hull = cv2.convexHull(cnt)
        cha = cv2.contourArea(hull)

        return area / cha


def main():
    dp = DetectApp()

    dp.upload_images()
    dp.process()


main()
