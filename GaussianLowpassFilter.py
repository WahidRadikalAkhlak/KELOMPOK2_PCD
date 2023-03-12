import cv2
import numpy as np

# load image
img = cv2.imread('gambar1.jpeg')

# convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# create Gaussian filter
filter_size = 11
sigma = 5
kernel = cv2.getGaussianKernel(filter_size, sigma)

# apply filter to image
filtered = cv2.filter2D(gray, -1, kernel)

# display original and filtered images
cv2.imshow('Original Image', gray)
cv2.imshow('Filtered Image', filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()