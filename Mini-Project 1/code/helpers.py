#-------------------Imports-----------------------------#
from matplotlib import pyplot as plt
import numpy as np
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale
#--------------------------------------------------------#
#--------------------Make blur filter--------------------#
def create_mean_filter(ksize:tuple):
    # Remember to assert that the length is odd
    assert ksize[0]%2!=0 and ksize[1]%2!=0
    tnc=ksize[0]*ksize[1]#Total number of cells
    K=np.ones(ksize)
    K=K*1/tnc
    return K
#--------------------------------------------------------#
#--------------------Convolution-------------------------#
def convolve(image, filter):
  r1, c1 = image.shape
  plt.imshow(image)
  r2, c2 = filter.shape
  a_padded = np.zeros((r1 + 2*r2 - 2, c1 + 2*c2 - 2))
  a_padded[r2 - 1:r2 -1 + r1, c2 - 1:c2 -1 + c1] = image
  output = np.zeros([r1 + r2 - 1, c1 + c2-1])
  r1, c1 = a_padded.shape
  r, c = output.shape
  sum = 0

  for i in range(r):
      for j in range(c):
          for k in range(r2):
              for l in range(c2):
                  sum += a_padded[i + k, j + l] * (filter[k, l])
          output[i, j] = sum
          sum = 0
      
  output = output[int(np.ceil((r2-1)/2)):int(np.ceil((r2-1)/2))+ image.shape[0], int(np.ceil((c2-1)/2)):int(np.ceil((c2-1)/2))+ image.shape[1]]
  return output
#--------------------------------------------------------#
def apply_conv(img,filter,channels):
  filtered_image=np.zeros(img.shape)
  for i in range(channels):
      filtered_image[...,i]= convolve(img[...,i], filter)
      plt.imshow(filtered_image)

  #filtered_image = filtered_image.astype(np.uint8)
  return filtered_image
#--------------------------------------------------------#

def my_imfilter(image: np.ndarray, filter: np.ndarray):
  """
  Your function should meet the requirements laid out on the project webpage.
  Apply a filter to an image. Return the filtered image.
  Inputs:
  - image -> numpy nd-array of dim (m, n, c) for RGB images or numpy nd-array of dim (m, n) for gray scale images
  - filter -> numpy nd-array of odd dim (k, l)
  Returns
  - filtered_image -> numpy nd-array of dim (m, n, c) or numpy nd-array of dim (m, n)
  Errors if:
  - filter has any even dimension -> raise an Exception with a suitable error message.
  """ 
  r1, c1,channels = image.shape
  k, l = filter.shape # filter's shape
  assert k % 2 != 0 and l % 2 != 0   #Return an error message for even filters, as their output is undefined
  filtered_image=apply_conv(image,filter,channels)
  return filtered_image

  

def gen_hybrid_image(image1: np.ndarray, image2: np.ndarray, cutoff_frequency: float):
  """
   Inputs:
   - image1 -> The image from which to take the low frequencies.
   - image2 -> The image from which to take the high frequencies.
   - cutoff_frequency -> The standard deviation, in pixels, of the Gaussian
                         blur that will remove high frequencies.

   Task:
   - Use my_imfilter to create 'low_frequencies' and 'high_frequencies'.
   - Combine them to create 'hybrid_image'.
  """

  assert image1.shape == image2.shape

  # Steps:
  # (1) Remove the high frequencies from image1 by blurring it. The amount of
  #     blur that works best will vary with different image pairs
  # generate a gaussian kernel with mean=0 and sigma = cutoff_frequency,
  # Just a heads up but think how you can generate 2D gaussian kernel from 1D gaussian kernel
  def create_gaussian_filter(ksize: tuple, sigma: float):
    x = np.linspace(-(ksize[0] - 1)/2, (ksize[0] - 1)/2, ksize[0])
    y = np.linspace(-(ksize[1] - 1)/2, (ksize[1] - 1)/2, ksize[1])
    xv, yv = np.meshgrid(x, y)
    f = np.exp(-1 * (xv + yv)**2 * (1 / (2 * sigma**2)))
    return f/f.sum()
  kernel = create_gaussian_filter(ksize = (3, 3), sigma = cutoff_frequency)
  
  # Your code here:
  low_frequencies = [my_imfilter(image1, kernel), my_imfilter(image2, kernel)]

  # (2) Remove the low frequencies from image2. The easiest way to do this is to
  #     subtract a blurred version of image2 from the original version of image2.
  #     This will give you an image centered at zero with negative values.
  # Your code here #
  high_frequencies = [image1 - my_imfilter(image1, kernel), image2 - my_imfilter(image2, kernel)]

  # (3) Combine the high frequencies and low frequencies
  # Your code here #
  hybrid_image = low_frequencies[0] + high_frequencies[1] # Replace with your implementation

  # (4) At this point, you need to be aware that values larger than 1.0
  # or less than 0.0 may cause issues in the functions in Python for saving
  # images to disk. These are called in proj1_part2 after the call to 
  # gen_hybrid_image().
  # One option is to clip (also called clamp) all values below 0.0 to 0.0, 
  # and all values larger than 1.0 to 1.0.
  # (5) As a good software development practice you may add some checks (assertions) for the shapes
  # and ranges of your results. This can be performed as test for the code during development or even
  # at production!

  return low_frequencies, high_frequencies, hybrid_image

def vis_hybrid_image(hybrid_image: np.ndarray):
  """
  Visualize a hybrid image by progressively downsampling the image and
  concatenating all of the images together.
  """
  scales = 5
  scale_factor = 0.5
  padding = 5
  original_height = hybrid_image.shape[0]
  num_colors = 1 if hybrid_image.ndim == 2 else 3

  output = np.copy(hybrid_image)
  cur_image = np.copy(hybrid_image)
  for scale in range(2, scales+1):
    # add padding
    output = np.hstack((output, np.ones((original_height, padding, num_colors),
                                        dtype=np.float32)))
    # downsample image
    cur_image = rescale(cur_image, scale_factor, mode='reflect')
    # pad the top to append to the output
    pad = np.ones((original_height-cur_image.shape[0], cur_image.shape[1],
                   num_colors), dtype=np.float32)
    tmp = np.vstack((pad, cur_image))
    output = np.hstack((output, tmp))
  return output

def load_image(path):
  return img_as_float32(io.imread(path))

def save_image(path, im):
  return io.imsave(path, img_as_ubyte(im.copy()))