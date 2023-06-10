import cv2
import numpy as np
import math

cam = "rtsp://Hackathon:SzentJozsef1@192.168.0.180:554/cam/realmonitor?channel=2&subtype=1"

# Load the camera
cap = cv2.VideoCapture(cam)

def show_picture(img):
    cv2.imshow('image', img)

def save_image(img, index):
    cv2.imwrite('test-' + str(index) + '.jpg', img)


def reduce_noise(img):
    return cv2.fastNlMeansDenoising(img, None, 10, 7, 21)

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
    
    b, g, r = cv2.split(img)

    # Apply Gaussian blur to each color channel
    blurred_b = cv2.GaussianBlur(b, (11, 11), 0)
    blurred_g = cv2.GaussianBlur(g, (11, 11), 0)
    blurred_r = cv2.GaussianBlur(r, (11, 11), 0)

    # Merge the blurred color channels back into an image
    img = cv2.merge([blurred_b, blurred_g, blurred_r])

    brightness = 90
    img = cv2.addWeighted(img, 1, img, 0, brightness)

    contrast = 500
    img = cv2.addWeighted(img, contrast, img, 0, int(round(255*(1-contrast)/2)))

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
    original_img = img
    #original_img = prepare_image(original_img)
    img = prepare_image(img)
    masks = np.array(['images/sector1_edge0 - Copy.jpg', 'images/sector2_edge0 - Copy.jpg', 
                      'images/sector3_edge0 - Copy.jpg', 'images/sector4_edge0 - Copy.jpg', 
                      'images/sector5_edge0 - Copy.jpg', 'images/sector6_edge0 - Copy.jpg', ])

    sectors = np.array([0, 0, 0, 0, 0, 0])
    sector_trigger = np.array([50, 100, 100, 80, 100, 100])
    index = 0


    for mask in masks:
        mask = cv2.imread(mask)
        mask = prepare_mask(mask)
        img2 = apply_mask(img, mask)
        print(detect_person(img2))
        if detect_person(img2) > sector_trigger[index]:
            sectors[index] = 1
            img2 = cv2.inRange(img2, (255, 255, 255), (255, 255, 255))
            img2 = cv2.morphologyEx(img2, cv2.MORPH_OPEN, None, iterations=2)
            contours, hierarchy = cv2.findContours(img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(original_img, contours, -1, (0, 255, 0), 2)
        index += 1

    show_picture(original_img)
    return sectors
    

res_cam_index = 0

while True:
    _, img = cap.read()

    if img is None or res_cam_index == 10:
        cap.release()
        cap = cv2.VideoCapture(cam)
        print("Rstarting camera loader!")
        res_cam_index = 0
        continue

    print(detect_active_sectors(img))

    res_cam_index += 1

    # sleep for 10000 ms
    cv2.waitKey(1000)