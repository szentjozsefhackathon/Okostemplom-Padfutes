# Code written by: Miszori Gergo, Urban Szabolcs, Boviz Daniel
import cv2
import numpy as np
import math
import switch
from os.path import exists
import os
import time

cam = "rtsp://Hackathon:SzentJozsef1@10.5.10.200:554/cam/realmonitor?channel=11&subtype=1"

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

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    
    # TODO: Fix this function
    rect = np.zeros((4, 2), dtype = "float32")
    
    
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    
    
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    
    
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    
    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    
    rect = order_points(pts) # not working properly
    (tl, tr, br, bl) = rect
    
    (tl, tr, br, bl) = pts
    
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    
    
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    
    # return the warped image
    return warped


# function to display the coordinates of 
# of the points clicked on the image  
def click_event(event, x, y, flags, params): 
  
    # checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN: 
  
        # displaying the coordinates 
        # on the Shell 
        print(x, ' ', y) 
  
        # displaying the coordinates 
        # on the image window 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(img, str(x) + ',' +
                    str(y), (x,y), font, 
                    1, (255, 0, 0), 2) 
        cv2.imshow('image', img) 
  
    # checking for right mouse clicks      
    if event==cv2.EVENT_RBUTTONDOWN: 
  
        # displaying the coordinates 
        # on the Shell 
        print(x, ' ', y) 
  
        # displaying the coordinates 
        # on the image window 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        b = img[y, x, 0] 
        g = img[y, x, 1] 
        r = img[y, x, 2] 
        cv2.putText(img, str(b) + ',' +
                    str(g) + ',' + str(r), 
                    (x,y), font, 1, 
                    (255, 255, 0), 2) 
        cv2.imshow('image', img) 

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
    detected_img = prepare_image(img)
    img = prepare_image(img)
    masks = np.array(['sect1.jpg', 'sect2.jpg', 'sect3.jpg',
                      'sect4.jpg', 'sect5.jpg', 'sect6.jpg' ])

    sectors = np.array([0, 0, 0, 0, 0, 0])
    sector_trigger = np.array([50, 120, 120, 80, 120, 120])
    index = 0

    for mask in masks:
        
        absolute_path = os.path.dirname(__file__)        
        mask = os.path.join(absolute_path, mask)
        file_exists = exists(mask)
        #print(file_exists)

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
        
        detected = detect_person(img2)
        if detect_person(img2) > sector_trigger[index]:
            sectors[index] = 1
            img2 = cv2.inRange(img2, (255, 255, 255), (255, 255, 255))
            img2 = cv2.morphologyEx(img2, cv2.MORPH_OPEN, None, iterations=2)
            contours, hierarchy = cv2.findContours(img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # Draw rectangles around the detected persons
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(detected_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            sectors[index] = 0

        print("Sector " + str(index + 1) + ": " + str(sectors[index]) + " (detect_person=" + str(detected) + ")")     
        index += 1
    
    #show_picture(original_img)
    path_to_tests = '/openhab_conf/scripts/Szent-Jozsef-Hackathon/tests/'
    now = str(int(time.time()))
    cv2.imwrite(path_to_tests + '/detected_' + now  + '_' + ''.join(map(str, sectors))  + '.jpg', detected_img)
    cv2.imwrite(path_to_tests + '/detected_' + now + '.jpg', original_img)
    
    return sectors
    

res_cam_index = 0

while True:
    _, img = cap.read()

    if img is None or res_cam_index == 10:
        cap.release()
        cap = cv2.VideoCapture(cam)
        res_cam_index = 0
        continue
    
    # active_sectors = detect_active_sectors(img)
    warped = four_point_transform(img, np.array([[249, 114], [721, 120], [1050, 451], [14, 589]]))
    show_picture(warped)


    

    # if active_sectors[0] == -1:
    #     print("The recogniser software can't work while the camera is in night mode!")
    # else:
    #     switch.switch(active_sectors)
    #     print(active_sectors)

    res_cam_index += 1

    # sleep for 10000 ms
    cv2.waitKey(10000)

    # # To get the four points of the rectangle from the original image
    # cv2.imshow('image', img)

    # cv2.setMouseCallback('image', click_event) 
  
    # # wait for a key to be pressed to exit 
    # cv2.waitKey(0) 
  
    # # close the window 
    # cv2.destroyAllWindows() 
    break
