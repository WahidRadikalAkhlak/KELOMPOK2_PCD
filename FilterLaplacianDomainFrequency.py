import cv2
import numpy as np
from numpy.fft import fft2, fftshift, ifft2

# Load the image
img = cv2.imread('gambar1.jpeg', cv2.IMREAD_GRAYSCALE)

# Calculate the FFT of the image
f = fftshift(fft2(img))

# Create a meshgrid for the Laplacian filter
rows, cols = img.shape
x = np.linspace(-cols/2, cols/2, cols)
y = np.linspace(-rows/2, rows/2, rows)
xx, yy = np.meshgrid(x, y)

# Create the Laplacian filter
radius = 10
L = -4 * np.pi ** 2 * (xx ** 2 + yy ** 2) / (rows ** 2 + cols ** 2)
H = np.exp(L * radius ** 2)

# Apply the Laplacian filter to the image in the frequency domain
filtered_f = f * H

# Calculate the inverse FFT of the filtered image
filtered_img = np.real(ifft2(fftshift(filtered_f)))

# Normalize the image
filtered_img = cv2.normalize(filtered_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Display the original image and the filtered image side by side
cv2.imshow('Original Image', img)
cv2.imshow('Laplacian Filtered Image', filtered_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
