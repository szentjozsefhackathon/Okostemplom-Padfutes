import cv2
import numpy as np

def detect_sitting_people(img):
    # Load the mask image
    mask = cv2.imread('detect-with-color-difference\mask.png')

    # Select only the (0, 155, 155) pixels on the mask
    m1 = np.all(mask == (0, 155, 155), axis=-1)

    # Calculate the spread of the pixels in the m1 array in the image
    spread = np.std(img[m1])

    # Print the spread
    print('Spread:', spread)

    # Show the image
    cv2.imshow('image', img)


# Load an image
img = cv2.imread('detect-with-color-difference\cam1.jpg')