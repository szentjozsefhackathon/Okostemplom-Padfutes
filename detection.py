# Code written by: Miszori Gergo, Urban Szabolcs, Boviz Daniel
import cv2
import numpy as np
import math
import switch

cam = "rtsp://Hackathon:SzentJozsef1@192.168.0.180:554/cam/realmonitor?channel=2&subtype=1"

# Load the camera
cap = cv2.VideoCapture(cam)

def show_picture(img):
    cv2.imshow('CAM2 - online footage', img)

def save_image(img, index):
    cv2.imwrite('test-' + str(index) + '.jpg', img)

def reduce_noise(img):
    return cv2.fastNlMeansDenoising(img, None, 1, 7, 21)

def apply_mask(img, mask):
    # Convert the img image to the same data type as the mask image
    img = img.astype(mask.dtype)

    # Create a new image with only the white pixels from the mask
    masked_image = cv2.bitwise_and(img, img, mask=mask)

    # Return the number of connected components
    return masked_image

def prepare_mask(mask):
    # Convert the mask to grayscale
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Innvert the mask
    mask = cv2.bitwise_not(mask)

    return mask

def prepare_image(img):
    img = reduce_noise(img)

    return img

def detect_person(img):
    # Remove white colors from the image
    img = cv2.inRange(img, (255, 255, 255), (255, 255, 255))
    
    # Remove noise
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, None, iterations=2)

    # Count the white pixels
    white_pixels = cv2.countNonZero(img)

    return white_pixels
    #return img

def detect_active_sectors(img):

    # If the img isn't rgb, return -1
    if len(img.shape) != 3:
        return np.array([-1, -1, -1, -1, -1, -1])

    original_img = img
    original_img = prepare_image(original_img)
    img = prepare_image(img)
    masks = np.array(['sect1.jpg', 'sect2.jpg', 'sect3.jpg',
                      'sect4.jpg', 'sect5.jpg', 'sect6.jpg' ])

    sectors = np.array([0, 0, 0, 0, 0, 0])
    sector_trigger = np.array([50, 120, 120, 80, 120, 120])
    index = 0

    for mask in masks:
        mask = cv2.imread(mask)
        mask = prepare_mask(mask)
        img2 = apply_mask(img, mask)

        brightness = 90
        contrast = 500

        if index == 0 or index == 3:
            brightness = 50
            contrast = 500
        
        if index == 1 or index == 4:
            brightness = 80
            contrast = 500

        img2 = cv2.addWeighted(img2, 1, img2, 0, brightness)
        img2 = cv2.addWeighted(img2, contrast, img2, 0, int(round(255*(1-contrast)/2)))

        print(detect_person(img2))
        if detect_person(img2) > sector_trigger[index]:
            sectors[index] = 1
            img2 = cv2.inRange(img2, (255, 255, 255), (255, 255, 255))
            img2 = cv2.morphologyEx(img2, cv2.MORPH_OPEN, None, iterations=2)
            contours, hierarchy = cv2.findContours(img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # Draw rectangles around the detected persons
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(original_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            sectors[index] = 0

            
        index += 1
    
    #show_picture(original_img)
    return sectors
    

res_cam_index = 0

while True:
    _, img = cap.read()

    if img is None or res_cam_index == 10:
        cap.release()
        cap = cv2.VideoCapture(cam)
        res_cam_index = 0
        continue
    
    active_sectors = detect_active_sectors(img)

    if active_sectors[0] == -1:
        print("The recogniser software can't work while the camera is in night mode!")
    else:
        switch.switch(active_sectors)
        print(active_sectors)

    res_cam_index += 1

    # sleep for 10000 ms
    cv2.waitKey(3000)
