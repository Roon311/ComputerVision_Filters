#---------------------------------Imports---------------------------------#
import os
import skimage
from skimage.transform import rescale
import numpy as np
from numpy import pi, exp, sqrt
import matplotlib.pyplot as plt
from PIL import Image 
import PIL 
import cv2
from helpers import load_image, save_image, my_imfilter,create_mean_filter,gen_hybrid_image,vis_hybrid_image,normalize_images
#-------------------------------------------------------------------------#

#----------------------------Make result Direct---------------------------#

resultsDir = 'results\\bonus\\part1\\'
if not os.path.exists(resultsDir):
    os.mkdir(resultsDir)
print("scikit-image version: {}".format(skimage.__version__))
#-------------------------------------------------------------------------#

#------------------------------Load the image-----------------------------#

test_image = load_image(r'data\zankyou.jpg')
print(test_image.shape)
print(type(test_image))
# test_image = rescale(test_image, 0.7, mode = 'reflect')
#-------------------------------------------------------------------------#
#-----------------------------Identity Filter-----------------------------#

identity_filter = np.asarray([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)#define the identity filter
identity_image = my_imfilter(test_image, identity_filter,fft=True)
print(identity_image.dtype)
identity_image=cv2.cvtColor(((identity_image+1)*255/2).astype(np.uint8), cv2.COLOR_BGR2RGB) 
identity_image_norm = cv2.normalize(identity_image, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imwrite(resultsDir+'fft_identity_image.jpg',identity_image_norm)
print('image saved')
#-------------------------------------------------------------------------#
#---------------------------------Box Filter------------------------------#

"""Small blur with a box filter This filter should remove some high frequencies."""
blur_filter = create_mean_filter((5,5))#create a 3*3 box filter
blur_filter /= np.sum(blur_filter, dtype=np.float32)  #divide the blur filter by the sum of of the filter
#print(blur_filter)
blur_image = my_imfilter(test_image, blur_filter,fft=True)
blur_image=cv2.cvtColor(((blur_image+1)*255/2).astype(np.uint8), cv2.COLOR_BGR2RGB) 
blur_image_norm = cv2.normalize(blur_image, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#cv2.imshow("frame2",blur_image_norm)
cv2.imwrite(resultsDir+'fft_blur_image.jpg', blur_image_norm)
#-------------------------------------------------------------------------#
#-------------------------------GaussianKernel----------------------------#

large_blur_filter=cv2.getGaussianKernel(7, 3)
large_blur_image = my_imfilter(test_image, large_blur_filter,fft=True);
print(large_blur_image)
large_blur_image=cv2.cvtColor(((large_blur_image*255)).astype(np.uint8), cv2.COLOR_BGR2RGB) 
large_blur_image_norm = cv2.normalize(large_blur_image, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imwrite(resultsDir+'fft_large_blur_image.jpg', large_blur_image_norm)
print('Large Blur saved')
#-------------------------------------------------------------------------#
#-------------------------------Sobel Filter------------------------------#

"""Oriented filter (Sobel operator)"""
sobel_filter = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)  # should respond to horizontal gradients
sobel_image = my_imfilter(test_image, sobel_filter,fft=True)

# 0.5 added because the output image is centered around zero otherwise and mostly black
sobel_image = np.clip(sobel_image+0.5, 0.0, 1.0)
plt.imshow(sobel_image)
done = save_image(resultsDir + os.sep + 'fft_sobel_image.jpg', sobel_image)
#-------------------------------------------------------------------------#

#-----------------------------Laplacian Filter----------------------------#

"""High pass filter (discrete Laplacian)"""
laplacian_filter = np.asarray([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
laplacian_image = my_imfilter(test_image, laplacian_filter,fft=True)

# added because the output image is centered around zero otherwise and mostly black
laplacian_image = np.clip(laplacian_image+0.5, 0.0, 1.0)
plt.figure(); plt.imshow(laplacian_image)
done = save_image(resultsDir + os.sep + 'fft_laplacian_image.jpg', laplacian_image)
#-------------------------------------------------------------------------#

#-----------------------------High-pass Filter----------------------------#

# High pass "filter" alternative
blur_image = blur_image.astype(np.float32)
blur_image /= 255.
high_pass_image = test_image - blur_image
high_pass_image = np.clip(high_pass_image+0.5, 0, 1)
plt.figure(); plt.imshow(high_pass_image)
done = save_image(resultsDir + os.sep + 'fft_high_pass_image.jpg', high_pass_image)
#-------------------------------------------------------------------------#
#--------------------------Hybrid image generation------------------------#
resultsDir = 'results\\bonus\\part2\\'
if not os.path.exists(resultsDir):
    os.mkdir(resultsDir)
image1 = load_image('data\dog.bmp')
image2 = load_image('data\cat.bmp')
cutoff_frequency = 11
low_frequencies, high_frequencies, hybrid_image = gen_hybrid_image(image1, image2, cutoff_frequency,fft=True)
vis = vis_hybrid_image(hybrid_image[0])
vis2 = vis_hybrid_image(hybrid_image[1])
to_normalize=[low_frequencies[0],high_frequencies[0],low_frequencies[1],high_frequencies[1],hybrid_image[1],hybrid_image[0],vis,vis2]
normalized=normalize_images(to_normalize)
#normalized[0]=cv2.cvtColor(((normalized[0])), cv2.COLOR_BGR2RGB) 
#normalized[2]=cv2.cvtColor(((normalized[2])), cv2.COLOR_BGR2RGB) 

names=['low_frequencies1.jpg','high_frequencies1.jpg','low_frequencies2.jpg','high_frequencies2.jpg','hybrid_image12.jpg','hybrid_image21.jpg','hybrid_image_scales12.jpg','hybrid_image_scales21.jpg']
for i,j in zip(normalized,names):
    im=cv2.cvtColor(((i)), cv2.COLOR_BGR2RGB) 
    cv2.imwrite(resultsDir+j, im)
#-------------------------------------------------------------------------#