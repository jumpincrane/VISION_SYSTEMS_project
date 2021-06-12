import cv2
import numpy as np
import os

def nothing(x):
    pass

def upload_images():
    images = []
    for root, dirs, files in os.walk('./images'):
        for f in files:
            if f.endswith('.jpg'):
                img_path = os.path.join(root, f)
                img = cv2.imread(img_path)
                images.append(img)
    return images

# Load imageno
images = upload_images()
for i in range(len(images)):
    image = images[i]
    image = cv2.resize(image, (int(image.shape[1] / 5), int(image.shape[0] / 5)))
    image = cv2.medianBlur(image, 3)
    image = cv2.convertScaleAbs(image, alpha=1.2, beta=1)
    # Create a window
    cv2.namedWindow('image')

    # Create trackbars for color change
    # Hue is from 0-179 for Opencv
    cv2.createTrackbar('HMin', 'image', 0, 179, nothing)
    cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
    cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMax', 'image', 0, 255, nothing)

    # Set default value for Max HSV trackbars


    # Initialize HSV min/max values
    dark_red = (np.array([165, 100, 40]), np.array([179, 255, 255]))
    color = dark_red
    hMin = color[0][0]
    sMin = color[0][1]
    vMin = color[0][2]
    hMax = color[1][0]
    sMax = color[1][1]
    vMax = color[1][2]
    cv2.setTrackbarPos('HMin', 'image', hMin)
    cv2.setTrackbarPos('SMin', 'image', sMin)
    cv2.setTrackbarPos('VSmin', 'image', vMin)
    cv2.setTrackbarPos('HMax', 'image', hMax)
    cv2.setTrackbarPos('SMax', 'image', sMax)
    cv2.setTrackbarPos('VMax', 'image', vMax)
    print(hMin, sMin, vMin)
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0

    while 1:
        # Get current positions of all trackbars
        hMin = cv2.getTrackbarPos('HMin', 'image')
        sMin = cv2.getTrackbarPos('SMin', 'image')
        vMin = cv2.getTrackbarPos('VMin', 'image')
        hMax = cv2.getTrackbarPos('HMax', 'image')
        sMax = cv2.getTrackbarPos('SMax', 'image')
        vMax = cv2.getTrackbarPos('VMax', 'image')

        # Set minimum and maximum HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        # Convert to HSV format and color threshold
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(image, image, mask=mask)

        # Print if there is a change in HSV value
        if((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
            print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax

        # Display result image
        cv2.imshow('image', result)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()