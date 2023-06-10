import cv2
import numpy as np
import math

cam = "rtsp://Hackathon:SzentJozsef1@192.168.0.180:554/cam/realmonitor?channel=2&subtype=1"
index = 10

# Load the camera
#cap = cv2.VideoCapture(cam)

def get_real_time_footage():
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

def create_edge_image(img):
    # Detect the edges
    edges = cv2.Canny(img, 100, 200)

    return edges

# Fast NINS denoise
def reduce_noise(img):
    return cv2.fastNlMeansDenoising(img, None, 10, 7, 21)

def apply_mask(img, mask):
    # Convert the img image to the same data type as the mask image
    img = img.astype(mask.dtype)

    # Create a new image with only the white pixels from the mask
    masked_image = cv2.bitwise_and(img, img, mask=mask)

    # Show the image
    show_picture(img)

    # Return the number of connected components
    return masked_image

def prepare_mask(mask):
    # Convert the mask to grayscale
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    #mask = cv2.dilate(mask, None, iterations=1)
    #mask = cv2.erode(mask, None, iterations=1)

    # Innvert the mask
    mask = cv2.bitwise_not(mask)

    return mask

def remove_horizontal_lines(img):
    # Detect the horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    detect_horizontal = cv2.morphologyEx(img, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    img = cv2.subtract(img, detect_horizontal)

    return img

def prepare_image(img):
    img = reduce_noise(img)
    
    brightness = 80
    img = cv2.addWeighted(img, 1, img, 0, brightness)

    contrast = 50
    img = cv2.addWeighted(img, contrast, img, 0, int(round(255*(1-contrast)/2)))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = create_edge_image(img)
    #img = cv2.dilate(img, None, iterations=1)
    #img = cv2.erode(img, None, iterations=1)
    return img

# Load an image
#img = cv2.imread('images/ulo/l22u.jpg')
img = cv2.imread('test-2.jpg')
base = cv2.imread('images/edges-0.jpg')
mask = cv2.imread('images/sector1_edge0.jpg')

# Prepare the image
img = prepare_image(img)
base = prepare_image(base)
mask = prepare_mask(mask)
# Apply the mask
#img = apply_mask(img, mask)

base = cv2.dilate(base, None, iterations=5)
img = cv2.subtract(img, base)

#img = remove_horizontal_lines(img)

# Show the image
show_picture(img)

# Wait until esc pressed
cv2.waitKey(0)

# Close the window
cv2.destroyAllWindows()
