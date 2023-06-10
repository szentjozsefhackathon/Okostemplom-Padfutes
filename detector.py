import cv2
import numpy as np

cam = "rtsp://Hackathon:SzentJozsef1@192.168.0.180:554/cam/realmonitor?channel=2&subtype=1"
index = 0

# Load the camera
cap = cv2.VideoCapture(cam)

def get_real_time_footage():
    _, img = cap.read()
    return img

def show_picture(img):
    cv2.imshow('image', img)

def save_image(img, index):
    cv2.imwrite('test-' + str(index) + '.jpg', img)

def create_edge_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the edges
    edges = cv2.Canny(gray, 100, 200)

    return edges

# Fast NINS denoise
def reduce_noise(img):
    return cv2.fastNlMeansDenoising(img, None, 10, 7, 21)

while True:
    # Read the frame
    img = get_real_time_footage()

    # Reduce noise
    img = reduce_noise(img)

    # Create an edge image
    edges = create_edge_image(img)

    # Merge the two picture
    img = cv2.addWeighted(img, 0.8, edges, 0.2, 0)

    # Show the image
    show_picture(img)

    # Save a picture if i press the 'p' button
    if cv2.waitKey(1) == ord('p'):
        save_image(img, index)
        index += 1
        print("Picture saved")

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

# Release the VideoCapture object
cap.release()
