import cv2
import numpy as np
from matplotlib import pyplot as plt

# Membaca citra
img = cv2.imread('gambar1.jpeg', 0)

# Menghitung DFT
dft = np.fft.fft2(img)

# Menggeser nol frekuensi ke tengah
dft_shift = np.fft.fftshift(dft)

# Menentukan radius cutoff dan nilai order filter
radius = 50
n = 2

# Membuat filter Butterworth highpass
rows, cols = img.shape
crow, ccol = rows//2, cols//2
u, v = np.meshgrid(np.arange(cols), np.arange(rows))
d_uv = np.sqrt((u - ccol)**2 + (v - crow)**2)
H = 1 / (1 + (d_uv/radius)**(2*n))

# Mengalikan filter dengan spektrum frekuensi
dft_shift_filtered = dft_shift * H

# Menggeser nol frekuensi kembali ke sudut kiri atas
dft_filtered = np.fft.ifftshift(dft_shift_filtered)

# Menghitung citra hasil filter dengan IDFT
img_filtered = np.fft.ifft2(dft_filtered)
img_filtered = np.abs(img_filtered)

# Menampilkan hasil
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_filtered, cmap='gray')
plt.title('Butterworth Highpass Filtered Image'), plt.xticks([]), plt.yticks([])
plt.show()