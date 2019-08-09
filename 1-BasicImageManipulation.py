
# Basic image manipulation
##-> convert nef image to rgb image(np.array)
##-> convert nef image to PIL image to use PIL pachages
##-> resize_image with scaling
##-> Save image to folder
##-> normalize image np.array to 0 to 1

import os 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as mp_image
import rawpy
import imageio
%matplotlib inline
   

# convert nef image to rgb (np.array)
def NEF2RGB(nef_image_path):
    import rawpy
    import imageio

    rp_image = rawpy.imread(nef_image_path)
    rgb = rp_image.postprocess()
    return rgb # np.array ([2300,2033, 3])

# convert nef file to PIL image files, so post image processing can use PIL package methods
def NEF2PIL(nef_image_path):
    import rawpy
    import imageio
    from PIL import Image
    rp_image = rawpy.imread(nef_image_path)
    rgb = rp_image.postprocess()
    rgb_pil = Image.fromarray(rgb)
    return rgb_pil

def PIL2RGB(pil_image):
    pil_rgb = np.array(pil_image)

    return pil_rgb

# Helper function to resize image of PIL image
def resize_image(src_img, size=(128,128), bg_color="white"): 
    from PIL import Image

    # rescale the image so the longest edge is the right size
    src_img.thumbnail(size, Image.ANTIALIAS)
    
    # Create a new image of the right shape
    new_image = Image.new("RGB", size, bg_color)
    
    # Paste the rescaled image onto the new background
    new_image.paste(src_img, (int((size[0] - src_img.size[0]) / 2), int((size[1] - src_img.size[1]) / 2)))
  
    # return the resized image
    return new_image

def save_image_PIL(img_PIL, image_folder_path, image_name):
    import os, shutil
    from PIL import Image

    # Delete the folder if it already exists
    if os.path.exists(image_folder_path):
            shutil.rmtree(image_folder_path)

    # Create the folder
    os.makedirs(image_folder_path)
    print("Ready to save images in", image_folder_path)

    image_name = "resized_baby.jpg"
    file_path = os.path.join(image_folder_path, file_name)

    # Save the image
    img_PIL.save(file_path, format="JPEG")
    
    return "image saved!"
## Other methods can save image: opencv, scikit, matplotlib, shown as following


##-->> use scikit-image imsave method
# Save RGB images (img_RGB) named as (image_name) to (image_folder_path)
def save_image_sk(img_RGB, image_folder_path, image_name):
    import skimage as sk
    from skimage import io as sk_io

    file_path = os.path.join(image_folder_path, image_name)

    # Save the image
    sk_io.imsave(fname=file_path, arr=img_RGB)
    
    return "Image saved!"

    ##-->> Use Opencv imwrite

# Save RGB images (img_RGB) named as (image_name) to (image_folder_path)
def save_image_cv(img_RGB, image_folder_path, image_name):
    import cv2

    file_path = os.path.join(image_folder_path, image_name)
    
    # Save the image
    cv2.imwrite(filename=file_path, img=img_RGB)

    return "Image saved!"
# Please note that the cv image is GBR format, instead of RGB

##-->> use matplotlib.pyplot imsave method
def save_image_plt(img_RGB, image_folder_path, image_name):
    import matplotlib.pyplot as plt

    file_path = os.path.join(image_folder_path, image_name)

    # Save the image
    plt.imsave(file_path, img_RGB) 

    return "Image saved!"

## Normalization of RGB np.array to 0 to 1
def imageNormalization(img_RGB):
    img_RGB_n = img_RGB * 1. / 255
    return img_RGB_n


# Test-- Read NEF images to RGB
NEF_path = "/Users/user7/Desktop/WeddingImageProcessing/data/image1/baby.nef"

# convert NEF image to RGB
nef_rgb = NEF2RGB(NEF_path)
print("Type of RGB images: ", type(nef_rgb))
print("Size of RGB images: ", nef_rgb.shape)

# Convert NEF image to PIL
nef_pil = NEF2PIL(NEF_path)
print("Type of PIL images: ", type(nef_pil))
print("Size of PIL images: ", nef_pil.size)

# Convert PIL image to RGB
nef_pil_rgb = PIL2RGB(nef_pil)
print("Type of NEF2PIL2RGB images: ", type(nef_pil_rgb))
print("Size of NEF2PIL2RGB images: ", nef_pil_rgb.shape)

# Plot test images
fig = plt.figure(figsize = (30,30))

a = fig.add_subplot(1,3,1)
img_plot_1 = plt.imshow(nef_rgb)
a.set_title("NEF2RGB Images")

a = fig.add_subplot(1,3,2)
img_plot_2 = plt.imshow(nef_pil)
a.set_title("NEF2PIL Images")

a = fig.add_subplot(1,3,3)
img_plot_3 = plt.imshow(nef_pil_rgb)
a.set_title("NEF2PIL2RGB Images")

plt.show()


