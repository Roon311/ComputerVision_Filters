import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import io, transform
from scipy.ndimage import convolve, correlate

# Load image and convert to grayscale
img = io.imread('writeup\RISDance.jpg', as_gray=True)

# Define filter sizes
filter_sizes = range(3, 16, 2)

# Define image sizes
image_sizes = [int(size) for size in np.logspace(np.log10(0.25e6), np.log10(8e6), num=10)]

# Initialize matrix of computation times
times_convolve = np.zeros((len(filter_sizes), len(image_sizes)))
times_correlate = np.zeros((len(filter_sizes), len(image_sizes)))

# Loop over filter sizes and image sizes and measure computation times
for i, size in enumerate(filter_sizes):
    filter_kernel = np.ones((size, size))
    for j, img_size in enumerate(image_sizes):
        # Resize image
        img_resized = transform.resize(img, (img_size // 1000, img_size // 1000), mode='constant')
        
        # Measure convolution computation time
        start_time = time.time()
        convolve(img_resized, filter_kernel)
        end_time = time.time()
        times_convolve[i, j] = end_time - start_time
        
        # Measure correlation computation time
        start_time = time.time()
        correlate(img_resized, filter_kernel)
        end_time = time.time()
        times_correlate[i, j] = end_time - start_time

# Plot matrix of results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(image_sizes, filter_sizes)
ax.plot_surface(X, Y, times_convolve, cmap='viridis', alpha=0.5)
ax.plot_surface(X, Y, times_correlate, cmap='plasma', alpha=0.5)
ax.set_xlabel('Image size (pixels)')
ax.set_ylabel('Filter size (pixels)')
ax.set_zlabel('Computation time (seconds)')
ax.set_title('Convolution (blue) and Correlation (orange) Computation Times')
plt.show()