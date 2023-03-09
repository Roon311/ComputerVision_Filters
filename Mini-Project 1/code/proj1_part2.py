#---------------------------------Imports---------------------------------#
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
from helpers import vis_hybrid_image, load_image, save_image, my_imfilter, gen_hybrid_image
#-------------------------------------------------------------------------#

""" Debugging tip: You can split your python code and print in between
to check if the current states of variables are expected or use a proper debugger."""

#------------------------------Load the image-----------------------------#
# Read images and convert to floating point format
image1 = load_image('data\dog.bmp')
image2 = load_image('data\cat.bmp')

# display the dog and cat images

plt.figure(figsize=(3,3))
plt.imshow((image1*255).astype(np.uint8))
plt.figure(figsize=(3,3))
plt.imshow((image2*255).astype(np.uint8))
#-------------------------------------------------------------------------#
#---------------------------------Notes---------------------------------#
'''
# For your write up, there are several additional test cases in 'data'.
# Feel free to make your own, too (you'll need to align the images in a
# photo editor such as Photoshop).
# The hybrid images will differ depending on which image you
# assign as image1 (which will provide the low frequencies) and which image
# you asign as image2 (which will provide the high frequencies)

## Hybrid Image Construction ##
# cutoff_frequency is the standard deviation, in pixels, of the Gaussian#
# blur that will remove high frequencies. You may tune this per image pair
# to achieve better results.
'''
#-------------------------------------------------------------------------#
#--------------------------Hybrid image generation------------------------#

cutoff_frequency = 7
low_frequencies, high_frequencies, hybrid_image = gen_hybrid_image(image1, image2, cutoff_frequency)
#-------------------------------------------------------------------------#

#----------------------------Visuaalize and Save--------------------------#
plt.figure()
plt.imshow((low_frequencies*255).astype(np.uint8))
plt.figure()
plt.imshow(((high_frequencies+0.5)*255).astype(np.uint8))
vis = vis_hybrid_image(hybrid_image)
plt.figure(figsize=(20, 20))
plt.imshow(vis)

save_image('../results/low_frequencies.jpg', low_frequencies)
save_image('../results/high_frequencies.jpg', high_frequencies+0.5)
save_image('../results/hybrid_image.jpg', hybrid_image)
save_image('../results/hybrid_image_scales.jpg', vis)
#-------------------------------------------------------------------------#
