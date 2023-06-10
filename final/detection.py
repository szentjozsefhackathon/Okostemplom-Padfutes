import cv2
import numpy as np
import math

cam = "rtsp://Hackathon:SzentJozsef1@192.168.0.180:554/cam/realmonitor?channel=2&subtype=1"

# Load the camera
cap = cv2.VideoCapture(cam)

def get_real_time_footage(cap):
    _, img = cap.read()

    if img is None:
        cap.release()
        cap = cv2.VideoCapture(cam)
        print("No image, restaring camera loader!")
        get_real_time_footage()

    return img

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
    
    brightness = 80
    img = cv2.addWeighted(img, 1, img, 0, brightness)

    contrast = 100
    img = cv2.addWeighted(img, contrast, img, 0, int(round(255*(1-contrast)/2)))

    return img

def detect_person(img):
    # Remove white colors from the image
    img = cv2.inRange(img, (128, 0, 0), (255, 255, 255))
    
    # Remove noise
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, None, iterations=2)

    # Count the white pixels
    white_pixels = cv2.countNonZero(img)

    return white_pixels
    #return img

def detect_active_sectors(img):
    img = prepare_image(img)
    masks = np.array(['images/sector1_edge0 - Copy.jpg', 'images/sector2_edge0 - Copy.jpg', 
                      'images/sector3_edge0 - Copy.jpg', 'images/sector4_edge0 - Copy.jpg', 
                      'images/sector5_edge0 - Copy.jpg', 'images/sector6_edge0 - Copy.jpg', ])

    sectors = np.array([0, 0, 0, 0, 0, 0])
    sector_trigger = np.array([50, 100, 100, 50, 100, 100])
    index = 0


    for mask in masks:
        mask = cv2.imread(mask)
        mask = prepare_mask(mask)
        img2 = apply_mask(img, mask)
        print(detect_person(img2))
        if detect_person(img2) > sector_trigger[index]:
            sectors[index] = 1
        index += 1

    return sectors

while True:
    img = get_real_time_footage(cap)

    show_picture(img)

    save_image(img, "fail")

    img = prepare_image(img)
    print(detect_active_sectors(img))

    # sleep for 10000 ms
    cv2.waitKey(1000)