from time import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import io, transform
from scipy.ndimage import convolve, correlate

def compute_duration(func, img, kernel):
    start = time()
    func(img, kernel)
    end = time()
    
    return end - start

image = io.imread('writeup\RISDance.jpg', as_gray = True) # Read the image as gray scale to reduce computations

filters = range(3, 16, 2) # Define filter sizes

images = np.linspace(0.25, 8, 10) * 1e6 # Define image sizes

# Constructing Time Matrices
times_convolve = np.zeros((len(filters), len(images)))
times_correlate = np.zeros_like(times_convolve) # To have the same shape as time convolve

# Loop over filter sizes and image sizes and measure computation times
for i, size in enumerate(filters):
    filter_kernel = np.ones((size, size))
    for j, img in enumerate(images):
        # Resize image
        img_resized = transform.resize(image, (img // 1000, img // 1000), mode='constant')
        
        # To compute the duration for convolvation and correlation
        times_convolve[i, j] = compute_duration(convolve, img_resized, filter_kernel)
        times_correlate[i, j] = compute_duration(correlate, img_resized, filter_kernel)

# Plot matrix of results
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection = '3d')
X, Y = np.meshgrid(images, filters)
ax.plot_surface(X, Y, times_convolve, cmap='viridis', alpha=0.5)
ax.set_xlabel('Image size (pixels)')
ax.set_ylabel('Filter size (pixels)')
ax.set_zlabel('Computation time (seconds)')
ax.set_title('Convolution Computation Times')
plt.show()

fig2 = plt.figure(figsize=(15, 15))
ax = fig2.add_subplot(111, projection = '3d')
X, Y = np.meshgrid(images, filters)
ax.plot_surface(X, Y, times_correlate, cmap='plasma', alpha=0.5)
ax.set_xlabel('Image size (pixels)')
ax.set_ylabel('Filter size (pixels)')
ax.set_zlabel('Computation time (seconds)')
ax.set_title('Correlation Computation Times')
plt.show()