import cv2
import numpy as np
from matplotlib import pyplot as plt

# Membaca citra
img = cv2.imread('gambar1.jpeg', 0)

# Menghitung DFT
dft = np.fft.fft2(img)

# Menggeser nol frekuensi ke tengah
dft_shift = np.fft.fftshift(dft)

# Menghitung magnitudo spektrum
magnitude_spectrum = 20*np.log(np.abs(dft_shift))

# Menampilkan hasil
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()