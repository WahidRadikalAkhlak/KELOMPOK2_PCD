import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('gambar1.jpeg', cv2.IMREAD_GRAYSCALE)

rows, cols = img.shape

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

D0 = 50 # cutoff frequency
n = 2 # order
H = np.zeros((rows, cols), np.float32)
for i in range(rows):
    for j in range(cols):
        D = np.sqrt((i - rows/2)**2 + (j - cols/2)**2)
        H[i,j] = 1 / (1 + (D0/D)**(2*n))

f_selective = fshift * H

f_ishift = np.fft.ifftshift(f_selective)
img_selective = np.fft.ifft2(f_ishift)
img_selective = np.abs(img_selective)

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_selective, cmap='gray')
plt.title('Selective Filtered Image'), plt.xticks([]), plt.yticks([])
plt.show()
