import numpy as np
import cv2
import os
import sys
import json
from math import *
from sklearn.cluster import KMeans
import time

class DetectApp:
    def __init__(self, images_path, result_path):
        self.images = []
        self.images_path = images_path
        self.result_path = result_path
        self.result = {}
        self.image_names = []

    def upload_images(self):
        for root, dirs, files in os.walk(self.images_path):
            for f in files:
                if f.endswith('.jpg'):
                    img_path = os.path.join(root, f)
                    img = cv2.imread(img_path)
                    self.images.append(img)
                    self.image_names.append(f)

    def process(self):
        start_process = time.time()
        for i in range(len(self.images)):
            print(self.image_names[i])
            start = time.time()
            img = self.images[i]
            objects = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            # =============
            bgr_img_not_float = cv2.pyrDown(cv2.pyrDown(img))
            # =================
            bgr_img = self.mask_color_adjust_color(img)
            hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv_img)
            # make threshold img
            ret, threshold = cv2.threshold(s, 0.43, 1.0, cv2.THRESH_BINARY)
            threshold = np.uint8(threshold)

            countours, hierarchy = cv2.findContours(threshold, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            for j, cnt in enumerate(countours):
                area = cv2.contourArea(cnt)
                if area > 400:
                    # mask found object
                    mask = np.zeros(bgr_img_not_float.shape[:2], np.uint8)
                    cv2.drawContours(mask, [cnt], -1, 255, -1)
                    # find center of mass with moments
                    M = cv2.moments(mask)
                    # find bounding rectangle to the contour and angle
                    rect = cv2.minAreaRect(cnt)
                    w = rect[1][0]
                    h = rect[1][1]
                    # calculate parameters of cnt
                    cnt_eccentricity = self.get_eccentricity(M)
                    # calculate ratio of areas ~ 2 = circle
                    r_w = w / 2
                    r_h = h / 2
                    circle_area_w = np.pi * r_w ** 2
                    circle_area_h = np.pi * r_h ** 2
                    extent = (circle_area_h + circle_area_w) / area

                    # find colors
                    # crop object
                    mask = cv2.erode(mask, None, iterations=2)
                    mask = cv2.bitwise_and(bgr_img_not_float, bgr_img_not_float, mask=mask)
                    x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(cnt)
                    mask = mask[y_rect:y_rect + h_rect, x_rect:x_rect + w_rect]
                    mask = np.reshape(mask, (mask.shape[0] * mask.shape[1], 3))
                    # define clusters
                    num_clusters = 3
                    clusters = KMeans(n_clusters=num_clusters)
                    clusters.fit(mask)
                    # make histogram
                    histogram = self.make_histogram(clusters)
                    combined = zip(histogram, clusters.cluster_centers_)
                    combined = sorted(combined, key=lambda x: x[0], reverse=True)

                    hsv_values = []
                    for index, rows in enumerate(combined):
                        bar, rgb, hsv = self.make_bar(100, 100, rows[1])
                        hsv_values.append(hsv)

                    for value in hsv_values:
                        if (0, 0, 0) <= value <= (0, 255, 255):
                            hsv_values.remove(value)

                    if 1.7 < extent < 2.21 and cnt_eccentricity > 0.62:
                        if (30, 40, 40) < hsv_values[0] < (75, 255, 255) and (30, 40, 40) < hsv_values[1] < (
                                75, 255, 255):
                            objects[9] += 1
                        elif (20, 40, 40) < hsv_values[0] < (29, 255, 255) and (20, 40, 40) < hsv_values[1] < (
                                29, 255, 255):
                            objects[10] += 1
                        elif (10, 40, 40) < hsv_values[0] < (19, 255, 255) and (10, 40, 40) < hsv_values[1] < (
                                19, 255, 255):
                            objects[8] += 1
                        elif (120, 40, 40) < hsv_values[0] < (179, 255, 255) and (120, 40, 40) < hsv_values[1] < (
                                179, 255, 255):
                            objects[7] += 1
                        elif (0, 40, 40) < hsv_values[0] < (9, 255, 255):
                            objects[6] += 1

                    elif 1.98 < extent < 3.7 and cnt_eccentricity < 0.62:
                        if (30, 40, 40) < hsv_values[0] < (75, 255, 255) and (30, 40, 40) < hsv_values[1] < (
                                75, 255, 255):
                            objects[3] += 1
                        elif (20, 40, 40) < hsv_values[0] < (29, 255, 255) and (20, 40, 40) < hsv_values[1] < (
                                29, 255, 255):
                            objects[4] += 1
                        elif (10, 40, 40) < hsv_values[0] < (19, 255, 255) and (10, 40, 40) < hsv_values[1] < (
                                19, 255, 255):
                            objects[2] += 1
                        elif (120, 40, 40) < hsv_values[0] < (179, 255, 255) and (120, 40, 40) < hsv_values[1] < (
                                179, 255, 255):
                            objects[1] += 1
                        elif (0, 40, 40) < hsv_values[0] < (9, 255, 255):
                            objects[0] += 1

                    elif extent > 3.7:
                        if (30, 40, 40) < hsv_values[0] < (75, 255, 255):
                            objects[12] += 1
                        elif (10, 40, 40) < hsv_values[0] < (19, 255, 255):
                            objects[13] += 1
                        elif (20, 40, 40) < hsv_values[0] < (30, 255, 255) or (0, 40, 40) < hsv_values[0] < (
                                9, 255, 255) or (170, 40, 40) < hsv_values[0] < (179, 255, 255):
                            objects[14] += 1

            # ======= results ============

            self.result[self.image_names[i]] = objects
            end = time.time()
            print('Ellapsed time for 1 image:', end - start)
        end_process = time.time()
        print('Processing 15 images: ', end_process - start_process)
        with open(self.result_path, 'w') as outfile:
            json.dump(self.result, outfile)

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
    def make_histogram(cluster):
        """
        Count the number of pixels in each cluster
        :param: KMeans cluster
        :return: numpy histogram
        """
        numLabels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
        hist, _ = np.histogram(cluster.labels_, bins=numLabels)
        hist = hist.astype('float32')
        hist /= hist.sum()
        return hist

    @staticmethod
    def make_bar(height, width, color):
        """
        Create an image of a given color
        :param: height of the image
        :param: width of the image
        :param: BGR pixel values of the color
        :return: tuple of bar, rgb values, and hsv values
        """
        bar = np.zeros((height, width, 3), np.uint8)
        bar[:] = color
        red, green, blue = int(color[2]), int(color[1]), int(color[0])
        hsv_bar = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv_bar[0][0]
        return bar, (red, green, blue), (hue, sat, val)

    @staticmethod
    def sort_hsvs(hsv_list):
        """
        Sort the list of HSV values
        :param hsv_list: List of HSV tuples
        :return: List of indexes, sorted by hue, then saturation, then value
        """
        bars_with_indexes = []
        for index, hsv_val in enumerate(hsv_list):
            bars_with_indexes.append((index, hsv_val[0], hsv_val[1], hsv_val[2]))
        bars_with_indexes.sort(key=lambda elem: (elem[1], elem[2], elem[3]))
        return [item[0] for item in bars_with_indexes]


def main():
    dp = DetectApp(sys.argv[1], sys.argv[2])
    dp.upload_images()
    dp.process()


main()
