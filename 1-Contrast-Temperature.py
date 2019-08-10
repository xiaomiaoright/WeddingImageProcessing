# Use python 
import os 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as mp_image
import rawpy
import imageio
%matplotlib inline

#Load the image
image_file = "/Users/user7/Desktop/WeddingImageProcessing/data/image1/baby.nef"
rp_image = rawpy.imread(image_file)
image = rp_image.postprocess() # rgb image, np.array
print(type(image))
print(image.shape)

# Display the image
plt.imshow(image)

## image normalization
# distribution of pixel values in the image
# Plot a histogram - we need to use ravel to "flatten" the 3 dimensions
plt.hist(image.ravel())
plt.show()

#cumulative distribution function (CDF) plot
plt.hist(image.ravel(), bins=255, cumulative=True)
plt.show()

# the result shows uneven distribution--> there is a lot of contrast in the image
# Ideally we should equalize the values in the images we want to analyse 
# to try to make our images more consistent 
# in terms of the shapes they contain irrespective of light levels.
# --> contrast stretching -> rescale the pixel values to ensure that all values between a very low and very high percentile (usually the 2nd percentile and 98th percentile) are mapped to the range 0 to 255 
# --> Histogram equalization -> creates a more uniform distribution -> uses the exposure.rescale_intensity and exposure.equalize_hist methods from the skimage package
from skimage import exposure
import numpy as np

# Contrast stretching
p2 = np.percentile(image, 5)
p98 = np.percentile(image, 90)
image_ct = exposure.rescale_intensity(image, in_range=(p2, p98))

image_eq = exposure.equalize_hist(image)

# Show the images
fig = plt.figure(figsize=(20, 12))

# Subplot for original image
a=fig.add_subplot(3,3,1)
imgplot = plt.imshow(image)
a.set_title('Original')

# Subplot for contrast stretched image
a=fig.add_subplot(3,3,2)
imgplot = plt.imshow(image_ct)
a.set_title('Contrast Stretched')

# Subplot for equalized image
a=fig.add_subplot(3,3,3)
imgplot = plt.imshow(image_eq)
a.set_title('Histogram Equalized')

# Subplots for histograms
a=fig.add_subplot(3,3,4)
imgplot = plt.hist(image.ravel())

a=fig.add_subplot(3,3,5)
imgplot = plt.hist(image_ct.ravel())

a=fig.add_subplot(3,3,6)
imgplot = plt.hist(image_eq.ravel())

# Subplots for CDFs

a=fig.add_subplot(3,3,7)
imgplot = plt.hist(image.ravel(), bins=255, cumulative=True)

a=fig.add_subplot(3,3,8)
imgplot = plt.hist(image_ct.ravel(), bins=255, cumulative=True)

a=fig.add_subplot(3,3,9)
imgplot = plt.hist(image_eq.ravel(), bins=255, cumulative=True)

plt.show()

## Change Temperature of PIL images
from PIL import Image
rp_image = rawpy.imread(image_file)
rgb = rp_image.postprocess()
rgb_pil = Image.fromarray(rgb)

kelvin_table = {
    1000: (255,56,0),
    1500: (255,109,0),
    2000: (255,137,18),
    2500: (255,161,72),
    3000: (255,180,107),
    3500: (255,196,137),
    4000: (255,209,163),
    4500: (255,219,186),
    5000: (255,228,206),
    5500: (255,236,224),
    6000: (255,243,239),
    6500: (255,249,253),
    7000: (245,243,255),
    7500: (235,238,255),
    8000: (227,233,255),
    8500: (220,229,255),
    9000: (214,225,255),
    9500: (208,222,255),
    10000: (204,219,255)}


def convert_temp(image, temp):
    r, g, b = kelvin_table[temp]
    matrix = ( r / 255.0, 0.0, 0.0, 0.0,
               0.0, g / 255.0, 0.0, 0.0,
               0.0, 0.0, b / 255.0, 0.0 )
    return image.convert('RGB', matrix)

rgb_pil_1000K = convert_temp(rgb_pil, 1000)
rgb_pil_2000K = convert_temp(rgb_pil, 2000)
rgb_pil_5000K = convert_temp(rgb_pil, 5000)
rgb_pil_7000K = convert_temp(rgb_pil, 7000)
rgb_pil_9500K = convert_temp(rgb_pil, 9500)
# Show the images
fig = plt.figure(figsize=(20, 12))

# Subplot for original image
a=fig.add_subplot(3,3,1)
imgplot = plt.imshow(image)
a.set_title('Original')

# Subplot for temperature adjusted image
a=fig.add_subplot(3,3,2)
imgplot = plt.imshow(rgb_pil_1000K)
a.set_title('1000K')

a=fig.add_subplot(3,3,3)
imgplot = plt.imshow(rgb_pil_2000K)
a.set_title('2000K')

a=fig.add_subplot(3,3,4)
imgplot = plt.imshow(rgb_pil_5000K)
a.set_title('5000K')

a=fig.add_subplot(3,3,5)
imgplot = plt.imshow(rgb_pil_7000K)
a.set_title('7000K')

a=fig.add_subplot(3,3,6)
imgplot = plt.imshow(rgb_pil_9500K)
a.set_title('9500K')

plt.show()

# Other editting actions of image to be finished
## Exposure
## Hilights
## Shadows