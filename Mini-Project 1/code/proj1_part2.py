#---------------------------------Imports---------------------------------#
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import os
from helpers import vis_hybrid_image, load_image, gen_hybrid_image,normalize_images
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

cutoff_frequency = 11
low_frequencies, high_frequencies, hybrid_image = gen_hybrid_image(image1, image2, cutoff_frequency)
#-------------------------------------------------------------------------#

#----------------------------Visuaalize and Save--------------------------#

vis = vis_hybrid_image(hybrid_image[0])
vis2 = vis_hybrid_image(hybrid_image[1])
#plt.figure()
#plt.imshow((low_frequencies[0]*255).astype(np.uint8))
#plt.figure()
#plt.imshow(((high_frequencies[1]+0.5)*255).astype(np.uint8))
#plt.figure(figsize=(20, 20))
#plt.imshow(vis)
print(hybrid_image[0])
to_normalize=[low_frequencies[0],high_frequencies[0],low_frequencies[1],high_frequencies[1],hybrid_image[1],hybrid_image[0],vis,vis2]
normalized=normalize_images(to_normalize)
names=['low_frequencies1.jpg','high_frequencies1.jpg','low_frequencies2.jpg','high_frequencies2.jpg','hybrid_image12.jpg','hybrid_image21.jpg','hybrid_image_scales12.jpg','hybrid_image_scales21.jpg']

resultsDir = 'results\\part2\\'
if not os.path.exists(resultsDir):
    os.mkdir(resultsDir)

print('start saving')
print(len(normalized))
for i,j in zip(normalized,names):
    im=cv2.cvtColor(((i)), cv2.COLOR_BGR2RGB) 
    cv2.imwrite(resultsDir+j, im)

print('saving successful')
#-------------------------------------------------------------------------#
