import numpy as np
import matplotlib.pyplot as plt
from scipy import fft, ifft

def ideal_lowpass_filter(image, cutoff_frequency):

    # Transformasi Fourier dari gambar
    frequency_domain = fft.fft2(image)

    # Menghitung jarak frekuensi dari setiap piksel dalam domain frekuensi
    freq_rows, freq_cols = frequency_domain.shape
    half_rows, half_cols = freq_rows // 2, freq_cols // 2
    row_indices = np.arange(freq_rows)
    col_indices = np.arange(freq_cols)
    col_indices[half_cols:] = col_indices[half_cols:] - freq_cols
    col_indices = np.fft.fftshift(col_indices)
    distance_matrix = np.sqrt((row_indices[:, np.newaxis] - half_rows)**2 + (col_indices[np.newaxis, :] - half_cols)**2)

    # Membuat filter dengan menggunakan persamaan Ideal Lowpass Filter
    filter = np.zeros((freq_rows, freq_cols))
    filter[distance_matrix <= cutoff_frequency] = 1

    # Mengaplikasikan filter pada domain frekuensi
    filtered_frequency_domain = frequency_domain * filter

    # Mengembalikan gambar hasil filter dengan transformasi balik Fourier
    return np.abs(fft.ifft(filtered_frequency_domain))

# Contoh penggunaan
from skimage import io, color

# Load gambar grayscale
image = color.rgb2gray(io.imread('gambar1.jpeg'))

# Aplikasikan Ideal Lowpass Filter dengan frekuensi cutoff 20 piksel
filtered_image = ideal_lowpass_filter(image, 400)

# Tampilkan gambar asli dan hasil filter
plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Gambar Asli')
plt.subplot(122), plt.imshow(filtered_image, cmap='gray'), plt.title('Hasil Filter')
plt.show()
