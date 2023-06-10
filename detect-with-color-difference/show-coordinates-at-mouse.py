# Load an image and show the coordinates of the mouse when clicked
import cv2
import numpy as np

# Load an image
img = cv2.imread('C:/Users/wasde/Documents/GitHub/Szent-Jozsef-Hackathon/detect-with-color-difference/cam1.jpg')

# Show the image
cv2.imshow('image', img)

# When the mouse is clicked, print the coordinates and in every four click create a rectangle
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ', ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        strXY = str(x) + ', ' + str(y)
        cv2.putText(img, strXY, (x, y), font, 0.5, (255, 255, 0), 2)
        cv2.imshow('image', img)

# Call the function
cv2.setMouseCallback('image', click_event)

# Wait for a key to be pressed to exit
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()
