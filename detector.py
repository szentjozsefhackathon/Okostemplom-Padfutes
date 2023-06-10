import cv2
import numpy as np
import math

cam = "rtsp://Hackathon:SzentJozsef1@192.168.0.180:554/cam/realmonitor?channel=2&subtype=1"
index = 10

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

def count_white_pixels(img, mask):
    # Grayscale the mask and the image
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Convert the img image to the same data type as the mask image
    img = img.astype(mask.dtype)

    # Create a new image with only the white pixels from the mask
    white_pixels = cv2.bitwise_and(img, img, mask=mask)

    # Display the image
    # show_picture(white_pixels)

    # Get the number of connected components
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(white_pixels, connectivity=8)

    # Decrease the number of connected components while the smallest component is smaller than 25 pixels
    if len(stats) > 1:
        while stats[np.argmin(stats[1:, -1]) + 1, -1] < 25:
            nb_components -= 1
            white_pixels[output == np.argmin(stats[1:, -1]) + 1] = 0
            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(white_pixels, connectivity=8)
            if len(stats) <= 1:
                break

    # Return the number of connected components
    return nb_components

def get_sitting_people(img):
    masks = np.array(["images/sector1_edge.jpg", "images/sector2_edge.jpg", "images/sector3_edge.jpg", 
                      "images/sector4_edge.jpg", "images/sector5_edge.jpg", "images/sector6_edge.jpg"])
    #masks = np.array(["images/sector3_edge.jpg"])
    sitting_people = np.array([])
    index = 0
    for mask in masks:
        # Load the mask
        mask = cv2.imread(mask)

        # Innvert the mask
        mask = cv2.bitwise_not(mask)

        # Delite the edges
        mask = cv2.erode(mask, None, iterations=3)

        # Count the white pixels
        white_pixels = count_white_pixels(img, mask)

        # If there are more than 1000 white pixels, then there is a person sitting
        print("Index: " + str(index) + " White pixels: " + str(white_pixels))
        if white_pixels > 50:
            sitting_people = np.append(sitting_people, index)

        index += 1

    return sitting_people

# Main loop
while True:
    # Read the frame
    #img = get_real_time_footage()
    img = cv2.imread("images/edges-1.jpg")

    if img is None:
        cap.release()
        cap = cv2.VideoCapture(cam)
        print("No image, restaring camera")
        continue

    # Reduce noise
    img = reduce_noise(img)

    # Create an edge image
    edges = create_edge_image(img)

    # Show the image
    show_picture(edges)

    # Save a picture if i press the 'p' button
    if cv2.waitKey(1) == ord('p'):
        save_image(img, index)
        index += 1
        print("Picture saved")

    if cv2.waitKey(1) == ord('q'):
        save_image(edges, index)
        index += 1
        print("Picture saved")

    # Get the sitting people
    sitting_people = get_sitting_people(edges)
    print("Sitting people: " + str(len(sitting_people)))

    # Wait 500 ms
    cv2.waitKey(1000)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

# Release the VideoCapture object
cap.release()
