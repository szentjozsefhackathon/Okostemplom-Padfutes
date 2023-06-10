import cv2
import numpy as np

# Get the footage of the links
cam1 = "rtsp://Hackathon:SzentJozsef1@192.168.0.180:554/cam/realmonitor?channel=2&subtype=1"
cam3 = "rtsp://Hackathon:SzentJozsef1@192.168.0.180:554/cam/realmonitor?channel=7&subtype=1"

# People cascade
sitting_people_cascade = cv2.CascadeClassifier('detect-with-ai/haarcascade_upperbody.xml')

# Load and display the image of cam1 and recognise all people and print their coordinates
cap = cv2.VideoCapture(cam1)

while True:
    # Read the frame
    _, img = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect people
    sitting_people = sitting_people_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw the rectangle around each person
    for (x, y, w, h) in sitting_people:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        print("Person at: " + str(x) + ", " + str(y))
        
    # Display
    cv2.imshow('img', img)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

# Release the VideoCapture object
cap.release()
