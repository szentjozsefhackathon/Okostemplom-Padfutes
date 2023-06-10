import numpy as np
import cv2

def detect_sitting_people(img):
    # Gray scale the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert the image into an array
    img = np.array(img)

    # There are sectors, every sector is given by 2 coordinates in a numpy array
    sectors = np.array([ [[250, 124], [432, 128]], [[240, 140], [430, 143]],
                              [[230, 158], [442, 160]], [[215, 180], [448, 185]],
                              [[197, 208], [454, 212]], [[181, 245], [463, 246]],
                              [[159, 288], [476, 289]], [[127, 335], [496, 342]],
                              [[104, 410], [520, 416]], [[65, 504], [551, 514]],
                              [[551, 122], [715, 134]], [[561, 135], [740, 150]],
                              [[577, 155], [770, 169]], [[599, 176], [801, 191]],
                              [[619, 197], [836, 218]], [[643, 229], [874, 251]],
                              [[680, 268], [918, 288]], [[723, 316], [971, 330]],
                              [[779, 380], [1024, 373]], [[838, 449], [1088, 430]]])
    
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
img = cv2.imread('detect-with-color-difference\cam1.jpg')

# Call the function
detect_sitting_people(img)

# Wait for a key to be pressed to exit
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()