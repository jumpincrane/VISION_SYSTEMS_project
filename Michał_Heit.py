import numpy as np
import cv2
import os


class DetectApp:
    def __init__(self):
        self.images = []
        self.objects = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0,
                        15: 0}
        self.green = (np.array([33, 80, 40]), np.array([75, 255, 255]))
        self.orange = (np.array([10, 100, 40]), np.array([19, 255, 255]))
        self.yellow = (np.array([21, 100, 40]), np.array([30, 255, 255]))
        self.light_red = (np.array([0, 100, 40]), np.array([9, 255, 255]))
        self.dark_red = (np.array([165, 100, 40]), np.array([179, 255, 255]))
        self.colors = (self.green, self.orange, self.yellow, self.light_red, self.dark_red)

    def upload_images(self):
        for root, dirs, files in os.walk('./images'):
            for f in files:
                if f.endswith('.jpg'):
                    img_path = os.path.join(root, f)
                    img = cv2.imread(img_path)
                    self.images.append(img)

    def process(self):
        for color in self.colors:
            for i in range(len(self.images)):
                print(f"Image_{i}")
                img = self.images[i]
                # colors
                bgr_img, c_cropped_img = self.mask_color_adjust_color(img, color[0], color[1])

                # make threshold img
                gr_img = cv2.cvtColor(c_cropped_img, cv2.COLOR_BGR2GRAY)
                ret, threshold = cv2.threshold(gr_img, 0, 255, cv2.THRESH_BINARY)

                # ============ DETECT CIRCLES =============
                circles = 0
                height, width = gr_img.shape
                circle_mask = np.zeros((height, width), np.uint8)
                circles, circle_mask = self.detect_circles(threshold, circle_mask)
                threshold = cv2.bitwise_and(threshold, circle_mask)
                if circles is not None:
                    self.objects[10] = len(circles[0, :])
                    circles = len(circles[0, :])
                else:
                    self.objects[10] = 0
                    circles = 0
                # ==========================================

                countours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                snake = 0
                bear = 0
                areas = []
                # ============ DETECT BEAR/SNAKE BASED ON AREA ===============
                for cnt in countours:
                    area = cv2.contourArea(cnt)
                    areas.append(area)

                max_area = max(areas)  # snake max
                low_area = 0.55 * max_area  # snake min
                bmax_area = 0.5 * max_area  # bear max
                bmin_area = 0.1 * max_area  # bear min
                noise_area = 0.07 * max_area  # noise

                for area in areas:
                    if low_area <= area <= max_area:
                        snake += 1
                        print("Waz", area)
                    elif bmin_area <= area <= bmax_area:
                        bear += 1
                        print("Mis", area)

                self.objects[13] = snake
                self.objects[4] = bear
                # =========================================================
                print(f"Mis:{bear}, Waz:{snake}, KOlka:{circles}")

                cv2.drawContours(threshold, countours, -1, (255, 255, 255), -1)
                cv2.imshow('d', bgr_img)
                # print(self.objects)
                cv2.waitKey()

    @staticmethod
    def mask_color_adjust_color(image, color_low, color_high):
        bgr_img = image
        bgr_img = cv2.resize(bgr_img, (int(bgr_img.shape[1] / 5), int(bgr_img.shape[0] / 5)))
        bgr_img = cv2.medianBlur(bgr_img, 3)
        bgr_img = cv2.convertScaleAbs(bgr_img, alpha=1.2, beta=1)

        hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

        curr_mask = cv2.inRange(hsv_img, color_low, color_high)
        g_cropped_img = cv2.bitwise_and(bgr_img, bgr_img, mask=curr_mask)

        return bgr_img, g_cropped_img

    @staticmethod
    def detect_circles(threshold, draw_img):
        dp = 1
        min_dst = 30
        param1 = 50
        param2 = 14
        minradius = 16
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


def main():
    dp = DetectApp()
    dp.upload_images()
    dp.process()


main()
