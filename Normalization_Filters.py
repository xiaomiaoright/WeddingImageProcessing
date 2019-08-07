import os 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as mp_image
import rawpy
import imageio
%matplotlib inline

#Load the image
image_file = "C:/Users/EyesHigh/Desktop/ImageProcessing/data/image1/baby.nef"
rp_image = rawpy.imread(image_file)
image = rp_image.postprocess()
print(type(image))
print(image.shape)

# Display the image
plt.imshow(image)

## image normalization
# distribution of pixel values in the image
# Plot a histogram - we need to use ravel to "flatten" the 3 dimensions
# Ideally, the image should have relatively even distribution of values, 
# indicating good contrast and making it easier to extract analytical information.
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
rgb_pil_5000K = convert_temp(rgb_pil, 7000)
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
imgplot = plt.imshow(rgb_pil_5000K)
a.set_title('7000K')

## Exposure
## Hilights
## Shadows


## Filters - Blur and Sharpen
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

# Required magic to display matplotlib plots in notebooks
%matplotlib inline

# Load the image from the source file
image_file = "C:/Users/EyesHigh/Desktop/ImageProcessing/data/image1/baby.nef"
rp_image = rawpy.imread(image_file)
rgb = rp_image.postprocess()
image = Image.fromarray(rgb)

blurred_image = image.filter(ImageFilter.BLUR)
sharpened_image = image.filter(ImageFilter.SHARPEN)

# Display it
fig = plt.figure(figsize=(16, 12))

# Plot original image
a=fig.add_subplot(1, 3, 1)
image_plot_1 = plt.imshow(image)
a.set_title("Original")

# Plot blurred image
a=fig.add_subplot(1, 3, 2)
image_plot_2 = plt.imshow(blurred_image)
a.set_title("Blurred")

# Plot sharpened image
a=fig.add_subplot(1, 3, 3)
image_plot_3 = plt.imshow(sharpened_image)
a.set_title("Sharpened")

plt.show()

## Filters -- create your own filters
my_kernel = (200, 50, -100,
             -50, 200, -50,
            -100, 50, 200)

filtered_image = image.filter(ImageFilter.Kernel((3,3), my_kernel))

# Display it
fig = plt.figure(figsize=(16, 12))

# Plot original image
a=fig.add_subplot(1, 2, 1)
image_plot_1 = plt.imshow(image)
a.set_title("Original")

# Plot filtered image
a=fig.add_subplot(1, 2, 2)
image_plot_2 = plt.imshow(filtered_image)
a.set_title("Custom Filter")

plt.show()


## Filters - detect edges
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

# Required magic to display matplotlib plots in notebooks
%matplotlib inline
# Load the image from the source file
image_file = "C:/Users/EyesHigh/Desktop/ImageProcessing/data/iamge2/badlands-moon.nef"
rp_image = rawpy.imread(image_file)
rgb = rp_image.postprocess()
image = Image.fromarray(rgb)

edges_image = image.filter(ImageFilter.FIND_EDGES)

# Display it
fig = plt.figure(figsize=(16, 12))

# Plot original image
a=fig.add_subplot(1, 2, 1)
image_plot_1 = plt.imshow(image)
a.set_title("Original")

# Plot filtered image
a=fig.add_subplot(1, 2, 2)
image_plot_2 = plt.imshow(edges_image)
a.set_title("Edges")

plt.show()

## Filters --Edge --  Sobel edge-detection algorithm
def edge_sobel(image):
    from scipy import ndimage
    import skimage.color as sc
    import numpy as np
    image = sc.rgb2gray(image) # Convert color image to gray scale
    dx = ndimage.sobel(image, 1)  # horizontal derivative
    dy = ndimage.sobel(image, 0)  # vertical derivative
    mag = np.hypot(dx, dy)  # magnitude
    mag *= 255.0 / np.amax(mag)  # normalize (Q&D)
    mag = mag.astype(np.uint8)
    return mag

sobel_image = edge_sobel(np.array(image))

# Display it
fig = plt.figure(figsize=(16, 12))

# Plot original image
a=fig.add_subplot(1, 3, 1)
image_plot_1 = plt.imshow(image)
a.set_title("Original")

# Plot PIL FIND_EDGES image
a=fig.add_subplot(1, 3, 2)
image_plot_2 = plt.imshow(edges_image)
a.set_title("Edges")

# Plot Sobel image
a=fig.add_subplot(1, 3, 3)
image_plot_2 = plt.imshow(sobel_image, cmap="gray") # Need to use a gray color map as we converted this to a grayscale image
a.set_title("Sobel")

plt.show()

## Closing and opening effect on real image
import os
from skimage import io as sk_io
import skimage.color as sk_col

image_file = "C:/Users/EyesHigh/Desktop/ImageProcessing/data/iamge2/badlands-moon.nef"
rp_image = rawpy.imread(image_file)
image = rp_image.postprocess()

# Convert to grayscale so we only have one channel
bw_image = sk_col.rgb2gray(image)

# Apply operations
eroded_image = sk_mm.erosion(bw_image)
dilated_image = sk_mm.dilation(bw_image)
closed_image = sk_mm.closing(bw_image)
opened_image = sk_mm.opening(bw_image)

# Display it
fig = plt.figure(figsize=(20,20))

# Plot original image
a=fig.add_subplot(5, 1, 1)
plt.imshow(bw_image, cmap="gray")
a.set_title("Original")

# Plot eroded image
a=fig.add_subplot(5, 1, 2)
plt.imshow(eroded_image, cmap="gray")
a.set_title("Eroded")

# Plot dilated image
a=fig.add_subplot(5, 1, 3)
plt.imshow(dilated_image, cmap="gray")
a.set_title("Dilated")

# Plot closed image
a=fig.add_subplot(5, 1, 4)
plt.imshow(closed_image, cmap="gray")
a.set_title("Closed")

# Plot opened image
a=fig.add_subplot(5, 1, 5)
plt.imshow(opened_image, cmap="gray")
a.set_title("Opened")

plt.show()

## Thresholding
import os
import numpy as np
from matplotlib import pyplot as plt
from skimage import io as sk_io, color as sk_col, morphology as sk_mm
from skimage.filters import threshold_mean

%matplotlib inline

# Load the image from the source file
image_file = "C:/Users/EyesHigh/Desktop/ImageProcessing/data/iamge2/badlands-moon.nef"
rp_image = rawpy.imread(image_file)
image = rp_image.postprocess()


# Convert to grayscale so we only have one channel
bw_image = sk_col.rgb2gray(image)

# Find the mean threshold value
mean_val = threshold_mean(bw_image)

# Threshold the image
binary_image = bw_image > mean_val

# Plot the thresholded image
fig = plt.figure(figsize=(3,3))
plt.imshow(binary_image, cmap="gray")
plt.title("Mean Threshold: " + str(mean_val))
plt.show()

## try_all_threshold
from skimage.filters import try_all_threshold

fig, ax = try_all_threshold(bw_image, figsize=(10, 10), verbose=False)
plt.show()


# Plot a histogram - we need to use ravel to "flatten" the 3 dimensions
plt.hist(bw_image.ravel())
plt.show()

## Our image is slightly bi-modal - in other words it has two peaks or maxima. Both the Minimum and Otsu thresholding techniques explit this by finding a value between the maxima:
from skimage.filters import threshold_minimum, threshold_otsu

# Apply Minimum thresholding
min_val = threshold_minimum(bw_image)
binary_image_min = bw_image > min_val

# Apply Otsu thresholding
otsu_val = threshold_otsu(bw_image)
binary_image_otsu = bw_image > otsu_val

# Display the thresholded images
fig = plt.figure(figsize=(12,12))

# Minimum
a=fig.add_subplot(1, 2, 1)
image_plot_1 = plt.imshow(binary_image_min, cmap="gray")
a.set_title("Min Threshold: " + str(min_val))

# Otsu
a=fig.add_subplot(1, 2, 2)
image_plot_2 = plt.imshow(binary_image_otsu, cmap="gray")
a.set_title("Otsu Threshold: " + str(otsu_val))

plt.show()

## clean up these sparse pixels:
from skimage.filters import threshold_triangle

# Apply Triangle thresholding
tri_val = threshold_triangle(bw_image)
binary_image_tri = bw_image > tri_val

# Apply erosion
closed_image_tri = sk_mm.closing(binary_image_tri)

fig = plt.figure(figsize=(12,12))

# Plot original image
a=fig.add_subplot(1, 2, 1)
plt.imshow(binary_image_tri, cmap="gray")
a.set_title("Triangle Thresholding")

# Plot eroded image
a=fig.add_subplot(1, 2, 2)
plt.imshow(closed_image_tri, cmap="gray")
a.set_title("Triangle Thresholding + Closing")

plt.show()