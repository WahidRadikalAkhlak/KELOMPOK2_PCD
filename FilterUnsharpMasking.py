import cv2
import numpy as np

img = cv2.imread('gambar1.jpeg')

kernel = np.array([[-1,-1,-1],
                   [-1, 9,-1],
                   [-1,-1,-1]])

sharp = cv2.filter2D(img, -1, kernel)

blurred = cv2.GaussianBlur(img, (5,5), 0)
()
unsharp_masked = cv2.addWeighted(sharp, 1.5, blurred, -0.5, 0)

cv2.imshow('Original Image', img)
cv2.imshow('Unsharp Masked Image', unsharp_masked)
cv2.waitKey(0)
cv2.destroyAllWindows()
