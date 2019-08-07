## Read NEF image files as np.array
## Convert np.array files to PIL images
## apply PIL packages to conduct images editing
import os 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as mp_image
import rawpy
import imageio
%matplotlib inline

src_folder = "C:/Users/EyesHigh/Desktop/ImageProcessing/data"
#fig = plt.figure(figsize=(12,12))
images = []

for root, folders, filenames in os.walk(src_folder):
    #image_num = 0
    num_folders = len(folders)
    for folder in sorted(folders):
        # keep an incrementing count of each image
        
        # Find the first image file in the folder
        file_name = os.listdir(os.path.join(root,folder))[0]
        # get the full path from the root folder
        file_path = os.path.join(root, folder, file_name)
        # Open the file using the matplotlib library
        rp_image = rawpy.imread(file_path)
        rgb = rp_image.postprocess()
        images.append(rgb)
        # Add the image to the figure (which will have a row for each folder, each containing one column for the image)
        #a = fig.add_subplot(num_folders, 1, image_num)
        # Add the image to plot
        #image_plot = plt.imshow(rgb)
        # Add a caaption with the folder name
        #a.set_title(folder)

        """
        print(rgb.shape) # rgb images file, np.array [1000,1000,3]
        """
        print(type(rgb))
        print(rgb.shape)
# Show plot
#plt.show()
# Set up a figure of an appropriate size
fig = plt.figure(figsize=(120, 120))

image_num = 0
num_images = len(images)
# loop through the images
for image_idx in range(num_images):
    # Keep an incrementing count of each image
    a=fig.add_subplot(1, num_images, image_idx+1)
    # Add the image to the plot
    image_plot = plt.imshow(images[image_idx])
    # Add a caption with the folder name
    a.set_title("Image " + str(image_idx+1))
        
# Show the plot
plt.show()

# install and import opencv and scikit-image
!pip install opencv-python
!pip install --upgrade scikit-image

from PIL import Image
import skimage as sk
from skimage import io as sk_io
import cv2
file_path1 = "C:/Users/EyesHigh/Desktop/ImageProcessing/data/image1/baby.nef"
pil_image = Image.open(file_path1)
cv_image = cv2.imread(file_path1)
sk_image = sk_io.imread(file_path1)
rp_image1 = rawpy.imread(file_path1)
rgb1 = rp_image1.postprocess()

print(rgb1.shape) # (2844, 4284, 3), original images
print(cv_image.shape) # (212, 320, 3), compressed images
print(sk_image.shape) # (212, 320, 3), compressed images
print(pil_image.shape) # pil image is not np.array o_h, o_w = pil_image.size

# PIL image convert to ny.array
pil_image_array = np.array(pil_image)
plt.imshow(pil_image_array)
type(pil_image_array)
print(pil_image_array.shape) # (212, 320, 3), compressed images

# Numpy array to PIL images
rgb1_pil = Image.fromarray(rgb1)
plt.imshow(rgb1_pil)
print(type(rgb1_pil))
rgb2 = np.array(rgb1_pil)
print(rgb2.shape) # (2844, 4284, 3), original image size

### Image Manipulation
## Image rotation
rotated_pil_image = pil_image.rotate(90, expand=1) # The expand parameter tells PIL to change the image dimenions to fit the rotated orientation
plt.imshow(rotated_pil_image)

from skimage import transform as sk_transform
rotated_sk_image = sk_transform.rotate(rgb1, 90, resize=True)
plt.imshow(rotated_sk_image)
print(rotated_sk_image.shape) # (4284, 2844, 3), original size
# NEF file can be read as np.array, and use skimage package to do transformation


## Flip images
import numpy as np
upended_cv_image_rgb = np.flip(rgb1, axis=0)
mirrored_cv_image_rgb = np.flip(rgb1, axis=1)
print(upended_cv_image_rgb.shape)
print(mirrored_cv_image_rgb.shape)

fig = plt.figure(figsize=(120, 120))

# Plot original image
a=fig.add_subplot(1, 3, 1)
image_plot_1 = plt.imshow(rgb1)
a.set_title("Original")

# Plot upended image
a=fig.add_subplot(1, 3, 2)
image_plot_2 = plt.imshow(upended_cv_image_rgb)
a.set_title("Flipped Vertically")

# Plot mirrored image
a=fig.add_subplot(1, 3, 3)
image_plot_3 = plt.imshow(mirrored_cv_image_rgb)
a.set_title("Flipped Horizontally")

plt.show()

## Resize Images
# PIL thumbnail method

from PIL import Image, ImageOps
import matplotlib.pyplot as plt
%matplotlib inline

# Get the PIL image size
o_h, o_w = rgb1_pil.size
print('Original size:', o_h, 'x', o_w) #Original size: 4284 x 2844

# We'll resize this so it's 150 pixels on its widest dimensions
target_size = (150,150)
resized_img = rgb1_pil.copy()
resized_img.thumbnail(target_size, Image.ANTIALIAS) # scale the image with ANTILIAS
n_h, n_w = resized_img.size
print('New size:', n_h, 'x', n_w) #New size: 150 x 99

# Show the original and resized images
# Create a figure
fig = plt.figure(figsize=(120, 120))

# Subplot for original image
a=fig.add_subplot(1,2,1)
imgplot = plt.imshow(rgb1_pil)
a.set_title('Before')

# Subplot for resized image
a=fig.add_subplot(1,2,2)
imgplot = plt.imshow(resized_img)
a.set_title('After')

plt.show()
#specified 150 pixels as the target size for both height and width, 
# the image was rescaled so that it's longest dimension (width) is set to 150 pixels, 
# and the height is resized proportionally to keep the right aspect ratio

from PIL import Image, ImageOps
import matplotlib.pyplot as plt
%matplotlib inline

# Get the image size
o_h, o_w = rgb1_pil.size
print('Original size:', o_h, 'x', o_w)

# We'll resize this so it's 150 x 150
target_size = (150,150)
new_img = rgb1_pil.resize(target_size)
n_h, n_w = new_img.size
print('New size:', n_h, 'x', n_w)

# Show the original and resized images
# Create a figure
fig = plt.figure(figsize=(12, 12))

# Subplot for original image
a=fig.add_subplot(1,2,1)
imgplot = plt.imshow(pil_image)
a.set_title('Before')

# Subplot for resized image
a=fig.add_subplot(1,2,2)
imgplot = plt.imshow(new_img)
a.set_title('After')

plt.show()

# use the thumbnail method to rescale the image 
# so that it's longest dimension fits the desired size; 
# and then we'll create a new "background" image of the right dimensions, 
# and then paste the rescaled thumbnail into the middle of the background
def resize_image(src_image, size=(200,200), bg_color="white"): 
        from PIL import Image, ImageOps 
        
        # resize the image so the longest dimension matches our target size
        src_image.thumbnail(size, Image.ANTIALIAS)
        
        # Create a new square background image
        new_image = Image.new("RGB", size, bg_color)
        
        # Paste the resized image into the center of the square background
        new_image.paste(src_image, (int((size[0] - src_image.size[0]) / 2), int((size[1] - src_image.size[1]) / 2)))
      
        # return the resized image
        return new_image

# Get the image size
o_h, o_w = rgb1_pil.size
print('Original size:', o_h, 'x', o_w)

# We'll resize this so it's 150 x 150 with black padding
target_size = (150,150)
pad_color = "black"
resized_img = resize_image(rgb1_pil.copy(), target_size, pad_color)
n_h, n_w = resized_img.size
print('New size:', n_h, 'x', n_w)

# Show the original and resized images
# Create a figure
fig = plt.figure(figsize=(120, 120))

# Subplot for original image
a=fig.add_subplot(1,2,1)
imgplot = plt.imshow(pil_image)
a.set_title('Before')

# Subplot for resized image
a=fig.add_subplot(1,2,2)
imgplot = plt.imshow(resized_img)
a.set_title('After')

plt.show()

## Save images
## to save PIL images
import os, shutil
image_folder = "C:/Users/EyesHigh/Desktop/ImageProcessing/my_images"
# Delete the folder if it already exists
if os.path.exists(image_folder):
        shutil.rmtree(image_folder)

# Create the folder
os.makedirs(image_folder)
print("Ready to save images in", image_folder)

file_name = "resized_baby.jpg"
file_path = os.path.join(image_folder, file_name)

# Save the image
resized_img.save(file_path, format="JPEG")
print("Image saved as ", file_path)

## use scikit-image imsave method
file_name2 = "rotated_baby2.jpg"
file_path = os.path.join(image_folder, file_name2)

# Save the image
sk_io.imsave(fname=file_path, arr=rgb2)
print("Image saved as", file_path)

## Use Opencv imwrite
file_name3 = "rotated_baby3.jpg"
file_path = os.path.join(image_folder, file_name3)

# Save the image
cv2.imwrite(filename=file_path, img=rgb2)
print("Image saved as", file_path)
# the cv image is different GBR instead of RGB

## use matplotlib.pyplot imsave method
file_name4 = "rotated_baby4.jpg"
file_path = os.path.join(image_folder, file_name4)

# Save the image
plt.imsave(file_path, rgb2)
print("Image saved as", file_path)

## when the images are all saved as JPEG files, we can load them
# Set up a figure of an appropriate size
fig = plt.figure(figsize=(12, 12))
file_names = os.listdir(image_folder)
img_num = 0
for file_name in file_names:
    file_path = os.path.join(image_folder, file_name)
    # Open the file using the matplotlib.image library
    image = mp_image.imread(file_path)
    # Add the image to the figure (which will have 1 row, a column for each filename, and a position based on its index in the file_names list)
    a=fig.add_subplot(1, len(file_names), file_names.index(file_name)+1)
    # Add the image to the plot
    image_plot = plt.imshow(image)
    # Add a caption with the file name
    a.set_title(file_name)
        
# Show the plot
plt.show()