import cv2
import numpy as np

img = cv2.imread('images\sectors.jpg')

image_array = np.array(img)
image_array = image_array.flatten()

file = open('test.txt', 'w')
for i in range(720):
    for j in range(1280):
        file.write(str(image_array[i*720+j]) + ' ')
    file.write('\n')
file.close()