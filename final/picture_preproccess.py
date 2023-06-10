import cv2
import numpy as np

def picture_preproccess(image):
    # Split the image into color channels
    b, g, r = cv2.split(image)

    # Apply Gaussian blur to each color channel
    blurred_b = cv2.GaussianBlur(b, (11, 11), 0)
    blurred_g = cv2.GaussianBlur(g, (11, 11), 0)
    blurred_r = cv2.GaussianBlur(r, (11, 11), 0)

    # Merge the blurred color channels back into an image
    blurred_image = cv2.merge([blurred_b, blurred_g, blurred_r])

    return blurred_image   

# Load the image
image = cv2.imread("test-2.jpg")

# Preprocess the image
result = picture_preproccess(image)

# Save the preprocessed image
cv2.imwrite("result.jpg", result)