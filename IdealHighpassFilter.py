import cv2
import numpy as np

# Load image
img = cv2.imread('gambar1.jpeg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply ideal highpass filter
d0 = 30 # cutoff frequency
rows, cols = gray.shape
crow, ccol = rows//2, cols//2
mask = np.ones((rows, cols), np.uint8)
mask[crow-d0:crow+d0, ccol-d0:ccol+d0] = 0
filtered = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(gray)) * mask))

# Normalize output
filtered = cv2.normalize(filtered.real, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Display original and filtered images
cv2.imshow('Original', gray)
cv2.imshow('Filtered', filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()
