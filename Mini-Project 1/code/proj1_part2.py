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
low_frequencies[0] = cv2.normalize(low_frequencies[0], dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
high_frequencies[0] = cv2.normalize(high_frequencies[0], dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
low_frequencies[1] = cv2.normalize(low_frequencies[1], dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
high_frequencies[1] = cv2.normalize(high_frequencies[1], dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
hybrid_image[1] = cv2.normalize(hybrid_image[1], dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
hybrid_image[0] = cv2.normalize(hybrid_image[0], dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
print(hybrid_image[0])
vis= cv2.normalize(vis, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
vis2= cv2.normalize(vis2, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)


print('start saving')
cv2.imwrite('results\\part2\\low_frequencies1.jpg', low_frequencies[0])
cv2.imwrite('results\\part2\\high_frequencies1.jpg', high_frequencies[0])
cv2.imwrite('results\\part2\\low_frequencies2.jpg', low_frequencies[1])
cv2.imwrite('results\\part2\\high_frequencies2.jpg', high_frequencies[1])
cv2.imwrite('results\\part2\\hybrid_image12.jpg', hybrid_image[0])
cv2.imwrite('results\\part2\\hybrid_image21.jpg', hybrid_image[1])
cv2.imwrite('results\\part2\\hybrid_image_scales12.jpg', vis)
cv2.imwrite('results\\part2\\hybrid_image_scales21.jpg', vis2)

#save_image('results/low_frequencies1.jpg', low_frequencies[0])
#save_image('results/high_frequencies1.jpg', high_frequencies[0])
#save_image('results/low_frequencies2.jpg', low_frequencies[1])
#save_image('results/high_frequencies2.jpg', high_frequencies[1])
#save_image('results/hybrid_image.jpg', hybrid_image)
#save_image('results/hybrid_image_scales.jpg', vis)
print('saving successful')
#-------------------------------------------------------------------------#
