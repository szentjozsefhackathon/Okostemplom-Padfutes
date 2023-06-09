# Get the footage of the links
cam1 = "rtsp://Hackathon:SzentJozsef1@192.168.0.180:554/cam/realmonitor?channel=2&subtype=1"
cam3 = "rtsp://Hackathon:SzentJozsef1@192.168.0.180:554/cam/realmonitor?channel=7&subtype=1"

import cv2
import numpy as np

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

index = 0

# To capture video from webcam
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(cam1)

while True:
    # Read the frame
    _, img = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Save a picture if i press the 'p' button
    if cv2.waitKey(1) == ord('p'):
        cv2.imwrite('test-' + str(index) + '.jpg', gray)
        index += 1
        print("Picture saved")

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Display
    cv2.imshow('img', gray)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

# Release the VideoCapture object
cap.release()