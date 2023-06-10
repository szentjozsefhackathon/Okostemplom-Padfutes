import numpy as np
import cv2

def detect_sitting_people(img):
    # Gray scale the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert the image into an array
    img = np.array(img)

    # There are sectors, every sector is given by 2 coordinates in a numpy array
    sectors_left = np.array([ [[228, 157], [441, 160]], [[10, 10], [20, 20]] ])
    sectors_right = np.array([ [[228, 157], [441, 160]], [[10, 10], [20, 20]] ])

    # Merge the sectors into one array
    sectors = np.array([sectors_left, sectors_right])

    # Create an array for the spreads
    spreads = np.array([])

    # The sectors are in a rectangle shape, calculate how spread is the pixel colors in the rectangle
    for sector in sectors:
        # Get the pixel colors in the sector
        sector_pixels = img[sector[0, 1]:sector[1, 1], sector[0, 0]:sector[1, 0]]

        # Calculate the spread
        spread = np.std(sector_pixels)

        # Add the spread to the spreads array
        spreads = np.append(spreads, spread)

    # Convert the array back to an RGB image
    img2 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Draw a red rectangle on the image
    for sector in sectors:
        cv2.rectangle(img2, (sector[0, 0], sector[0, 1]), (sector[1, 0], sector[1, 1]), (0, 0, 255), 2)

    # Show the image
    cv2.imshow('image', img2)

    # Print the spread
    for i in range(len(spreads)):
        print('Spread of sector', i, ':', spreads[i])

# Load an image
img = cv2.imread('C:/Users/wasde/Documents/GitHub/Szent-Jozsef-Hackathon/detect-with-color-difference/cam1.jpg')

# Call the function
detect_sitting_people(img)

# Wait for a key to be pressed to exit
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()