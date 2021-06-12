import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

import sys
import json


class DetectApp:
    def __init__(self):
        self.images = []
        self.image_names = []
        self.result = {}
        self.model = 'structured_edge/model.yml'
        self.objects = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def upload_images(self):
        for root, dirs, files in os.walk("./images"):
            for f in files:
                if f.endswith('.jpg'):
                    img_path = os.path.join(root, f)
                    img = cv2.imread(img_path)
                    self.images.append(img)
                    self.image_names.append(f)

    def process(self):
        for i in range(len(self.images)):

            print(self.image_names[i])
            img = self.images[i]
            # ========= CORRECTIONS ================
            img = img / np.float32(255)
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
            # ====================================

            cv2.namedWindow('img', cv2.WINDOW_NORMAL)
            cv2.imshow('img', img)
            cv2.waitKey()

    @staticmethod
    def detect_circles(threshold, draw_img):
        dp = 1
        min_dst = 200
        param1 = 50
        param2 = 20
        minradius = 25
        maxradius = 50

        circles = cv2.HoughCircles(threshold, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dst, param1=param1,
                                   param2=param2, minRadius=minradius,
                                   maxRadius=maxradius)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for j in circles[0, :]:
                center = (j[0], j[1])
                cv2.circle(draw_img, center, 1, (0, 100, 100), -1)
                radius = j[2]
                cv2.circle(draw_img, center, radius, (255, 0, 255), -1)

        draw_img = cv2.bitwise_not(draw_img)
        return circles, draw_img

    def nothing(x):
        pass


def main():
    dp = DetectApp()
    dp.upload_images()
    dp.process()


main()
