import numpy as np
import cv2

def detect_sitting_people(img):
    # Gray scale the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Show the image
    cv2.imshow('image', img)

    # Convert the image into an array
    img = np.array(img)

    # There are sectors, every sector is given by 2 coordinates in a numpy array
    #sector = np.array([[125, 337], [497, 348]])
    sector = np.array([[1, 1], [500, 500]])

    # The sectors are in a rectangle shape, calculate how spread is the pixel colors in the rectangle
    supread_sector_1 = np.std(img[sector[0][0]:sector[1][0], sector[0][1]:sector[1][1]], dtype=int)

    # Print the spread
    print(supread_sector_1)

# Load an image
img = cv2.imread('C:/Users/wasde/Documents/GitHub/Szent-Jozsef-Hackathon/detect-with-color-difference/cam1.jpg')

# Call the function
detect_sitting_people(img)

# Wait for a key to be pressed to exit
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()