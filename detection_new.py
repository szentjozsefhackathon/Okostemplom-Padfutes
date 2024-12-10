# Code written by: Miszori Gergo, Urban Szabolcs, Boviz Daniel
import cv2
import numpy as np
import math
import switch
from os.path import exists
import os
import time

CAPTURE_SECONDS = 20
LEARNING_RATE = -1 
MOTION_THRESHOLD = 10  
# if empty: use video stream; if not, use an avi file
#INPUT_FILE = "testvideo.avi"
INPUT_FILE = ""


path_to_tests = '/openhab_conf/scripts/Szent-Jozsef-Hackathon/tests/'

masks = np.array(['sect1full.jpg', 'sect2full.jpg', 'sect3full.jpg',
                      'sect4full.jpg', 'sect5full.jpg', 'sect6full.jpg' ])

fgbg = cv2.createBackgroundSubtractorMOG2()

# Load the camera
if INPUT_FILE == "":
    cam = "rtsp://Hackathon:SzentJozsef1@10.5.10.200:554/cam/realmonitor?channel=11&subtype=1"
    cap = cv2.VideoCapture(cam)


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
   
    now = str(time.strftime("%d%b%Y_%H%M%S", time.gmtime()))
    print( "write imgs " + now )
    cv2.imwrite(path_to_tests2 + '/detected_' + now  + '_' + ''.join(map(str, sectors))  + '.jpg', detected_img)
    cv2.imwrite(path_to_tests2 + '/detected_' + now + '.jpg', original_img)
    
    return sectors
    

res_cam_index = 0



#save video
now = str(time.strftime("%d%b%Y_%H%M%S", time.localtime()))
fourcc = cv2.VideoWriter_fourcc(*'XVID')

back_win = "background"
motion_win = "motion"

#cv2.namedWindow(back_win, cv2.WINDOW_NORMAL)
cv2.namedWindow(motion_win, cv2.WINDOW_NORMAL)

while True:

    now = str(time.strftime("%d%b%Y_%H%M%S", time.localtime()))
    fileVid = 'origVideo'+now+'.avi'
    fileMotion = 'motionVideo'+now+'.jpg'

    # 1. create video for detection
    if INPUT_FILE == "":

        start_time = time.time()
        vidMotion = cv2.VideoWriter( fileVid,fourcc, 20.0, (1280,720))

        while( int(time.time() - start_time) < CAPTURE_SECONDS ):
            ret, frame = cap.read()
            if ret==True:
                vidMotion.write(frame)
            else:
                break
                
        vidMotion.release()
    else:
        fileVid = INPUT_FILE
        fileMotion = INPUT_FILE[:-4] + ".jpg"
        
    vidInput = cv2.VideoCapture(fileVid)    

    #create sumImage
    ret, frameSum = vidInput.read()
    frameSum = cv2.cvtColor(frameSum, cv2.COLOR_BGR2GRAY)  
    frameSum = cv2.GaussianBlur(frameSum, (11, 11), 0)        
    frameSum = fgbg.apply(frameSum, LEARNING_RATE)

    # 2. create motion image from video
    while True:
        ret, frame = vidInput.read()
        if ret==True:
           
            # convert to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
            # blur - noise reduction
            #gray_frame = cv2.fastNlMeansDenoising(gray_frame, None, 1, 7, 21)        
            gray_frame = cv2.GaussianBlur(gray_frame, (11, 11), 0)        
            # background subtract
            fg_mask = fgbg.apply(gray_frame, LEARNING_RATE)
            #Get background
            background_mask = fgbg.getBackgroundImage()
            # remove noise
            fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]
            # create sum image
            cv2.add( frameSum, fg_mask)
            
        else:
            break
            
    # 3. display
    # empty greyscale image
    emptyimage = np.full((720, 1280), 0, dtype=np.uint8)
    final = frameSum
    
    active_sectors = np.array([0, 0, 0, 0, 0, 0])
    index = 0

    # mask sumFrame
    print(str(now), flush=True)

    for mask in masks:
        mask = cv2.imread(mask)
        mask = prepare_mask(mask)
        frameSum_masked = apply_mask(frameSum, mask)
        if cv2.countNonZero(frameSum_masked) > MOTION_THRESHOLD:
            frameOut = cv2.addWeighted(frameSum_masked, 0.8, mask, 0.2, 0.0)
            active_sectors[index] = 1
            print("sector"+str(index+1)+": 1", flush=True)
        else:
            frameOut = emptyimage
            active_sectors[index] = 0
            print("sector"+str(index+1)+": 0", flush=True)

        final = final | frameOut
        index += 1    
    
    switch.switch(active_sectors)

    cv2.imshow(motion_win, final) 
    cv2.imwrite(fileMotion, final)
    
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
vidMotion.release()
cap.release()
cv2.destroyAllWindows()