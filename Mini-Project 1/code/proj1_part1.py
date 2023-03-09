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
from helpers import load_image, save_image, my_imfilter,create_mean_filter
#-------------------------------------------------------------------------#

#----------------------------Make result Direct---------------------------#

resultsDir = 'results\\'
if not os.path.exists(resultsDir):
    os.mkdir(resultsDir)
print("scikit-image version: {}".format(skimage.__version__))
#-------------------------------------------------------------------------#

#------------------------------Load the image-----------------------------#

test_image = load_image(r'data\dog.bmp')
print(test_image.shape)
print(type(test_image))
# test_image = rescale(test_image, 0.7, mode = 'reflect')
#-------------------------------------------------------------------------#
'''
#-----------------------------Identity Filter-----------------------------#

#cv2.imshow("test_image",test_image)#view the test image 
identity_filter = np.asarray([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)#define the identity filter
identity_image = my_imfilter(test_image, identity_filter)
print(identity_image.dtype)
identity_image=cv2.cvtColor(((identity_image+1)*255/2).astype(np.uint8), cv2.COLOR_BGR2RGB) 
#cv2.imshow("frame1",identity_image)
identity_image_norm = cv2.normalize(identity_image, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imwrite('results\identity_image.jpg',identity_image_norm)
print('image saved')
#-------------------------------------------------------------------------#

#---------------------------------Box Filter------------------------------#
'''
"""Small blur with a box filter This filter should remove some high frequencies."""
blur_filter = create_mean_filter((5,5))#create a 3*3 box filter
blur_filter /= np.sum(blur_filter, dtype=np.float32)  #divide the blur filter by the sum of of the filter
#print(blur_filter)
blur_image = my_imfilter(test_image, blur_filter)
blur_image=cv2.cvtColor(((blur_image+1)*255/2).astype(np.uint8), cv2.COLOR_BGR2RGB) 
blur_image_norm = cv2.normalize(blur_image, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#cv2.imshow("frame2",blur_image_norm)
#cv2.imwrite('results\\blur_image.jpg', blur_image_norm)
#-------------------------------------------------------------------------#
'''
#-------------------------------GaussianKernel----------------------------#
"""Large blur:This blur would be slow to do directly, so we instead use the fact that Gaussian blurs are separable and blur sequentially in each direction."""
# generate a gaussian kernel with any parameters of your choice. you may only in this case use a function
# from any library to generate the kernel such as: cv2.getGaussianKernel() then use the kernel to check your
# my_imfilter() implementation
# https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#Mat%20getGaussianKernel(int%20ksize,%20double%20sigma,%20int%20ktype)
large_blur_filter=cv2.getGaussianKernel(7, 3)
large_blur_image = my_imfilter(test_image, large_blur_filter);
#cv2.imshow("large blur",large_blur_image)
print(large_blur_image)
large_blur_image=cv2.cvtColor(((large_blur_image*255)).astype(np.uint8), cv2.COLOR_BGR2RGB) 
#cv2.imshow("large blur2",large_blur_image)
large_blur_image_norm = cv2.normalize(large_blur_image, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#cv2.imshow("frame3",large_blur_image_norm)
cv2.imwrite('results\\large_blur_image.jpg', large_blur_image_norm)
print('Large Blur saved')
#-------------------------------------------------------------------------#
#------------------------------Naive Large Blur---------------------------#



## Slow (naive) version of large blur
# import time
# large_blur_filter = np.dot(large_1d_blur_filter, large_1d_blur_filter.T)
# t = time.time()
# large_blur_image = my_imfilter(test_image, large_blur_filter);
# t = time.time() - t
# print('{:f} seconds'.format(t))
##
#-------------------------------------------------------------------------#

"""
Oriented filter (Sobel operator)
"""
sobel_filter = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)  # should respond to horizontal gradients
sobel_image = my_imfilter(test_image, sobel_filter)

# 0.5 added because the output image is centered around zero otherwise and mostly black
sobel_image = np.clip(sobel_image+0.5, 0.0, 1.0)
plt.imshow(sobel_image)
done = save_image(resultsDir + os.sep + 'sobel_image.jpg', sobel_image)


"""
High pass filter (discrete Laplacian)
"""
laplacian_filter = np.asarray([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
laplacian_image = my_imfilter(test_image, laplacian_filter)

# added because the output image is centered around zero otherwise and mostly black
laplacian_image = np.clip(laplacian_image+0.5, 0.0, 1.0)
plt.figure(); plt.imshow(laplacian_image)
done = save_image(resultsDir + os.sep + 'laplacian_image.jpg', laplacian_image)
'''
# High pass "filter" alternative
high_pass_image = test_image - blur_image
print(high_pass_image)
high_pass_image=cv2.cvtColor(((high_pass_image*255)).astype(np.uint8), cv2.COLOR_BGR2RGB) 
print(high_pass_image)
high_pass_image = np.clip(high_pass_image+125, 0, 255)
cv2.imshow("frame4",high_pass_image)
cv2.imwrite('results\\high_pass_image.jpg', high_pass_image)
#print(high_pass_image)
#plt.figure(); plt.imshow(high_pass_image)
#done = save_image(resultsDir + os.sep + 'high_pass_image.jpg', high_pass_image)
