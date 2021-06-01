import numpy as np
import cv2
import scipy
import os
import skimage


class DetectApp:
    def __init__(self):
        self.images = []
        self.images_gs = []

    def upload_images(self):
        for root, dirs, files in os.walk('./images'):
            for f in files:
                if f.endswith('.jpg'):
                    img_path = os.path.join(root, f)
                    img = cv2.imread(img_path)
                    img_gs = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    self.images.append(img)
                    self.images_gs.append(img_gs)

    def show_images(self):

        for image in self.images_gs:
            cv2.imshow("img", image)
            cv2.waitKey(300)

stop_data = cv2.CascadeClassifier('stop_data.xml')
def main():
    dp = DetectApp()
    dp.upload_images()


main()
