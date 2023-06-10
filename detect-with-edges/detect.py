import cv2
import numpy as np

def detect_edges_on_the_image_and_create_an_edge_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the edges
    edges = cv2.Canny(gray, 100, 200)

    # Show the image
    cv2.imshow('image', edges)

    # Save the image
    cv2.imwrite('detect-with-edges\cam1-edges.jpg', edges)

    return edges

# Load an image
img = cv2.imread('detect-with-edges\cam1.jpg')

# Detect edges on the image and create an edge image
edges = detect_edges_on_the_image_and_create_an_edge_image(img)

# Wait for a key
cv2.waitKey(0)

# Destroy all the windows
cv2.destroyAllWindows()