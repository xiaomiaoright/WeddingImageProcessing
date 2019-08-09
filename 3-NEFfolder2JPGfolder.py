# given the NEF files, read through the NEF files, and create features array
# 1. save NEF files to RGB files to a new folder
# 2. can set target image size to resize all the images

import rawpy
import imageio
import skimage as sk
from skimage import io as sk_io
import os

# Conver NEF file to PIL
def NEF2PIL(nef_image_path):
    import rawpy
    import imageio
    from PIL import Image
    rp_image = rawpy.imread(nef_image_path)
    rgb = rp_image.postprocess()
    rgb_pil = Image.fromarray(rgb)
    return rgb_pil

# Resize PIL image to a fixed image size
def resize_PIL(img_PIL, target_size=(200, 200)):
    from PIL import Image, ImageOps

    img_PIL_resized = img_PIL.resize(target_size)

    return img_PIL_resized

# Save PIL images as JPG
def save_image_PIL(rgb_pil, image_folder_path, image_name):
    import os, shutil
    from PIL import Image

    file_path = os.path.join(image_folder_path, image_name)
    
    # Save the image
    rgb_pil.save(file_path, format="JPEG")
    
    return None


def NEFfolder2JPGfolder(NEF_folder_path, JPG_folder_path, target_size=(200,200)):
    import os 
    # go through all XMP files in folder
    NEF_files = os.listdir(NEF_folder_path)

    
    # Get the images and show the predicted classes
    for file_idx in range(len(NEF_files)):
        # set the JPG iamge name
        image_name = NEF_files[file_idx]
        image_name = image_name.split(".")[0]
        image_name = image_name + ".jpg"

        # get the NEF image file path
        NEF_path = os.path.join(NEF_folder_path, NEF_files[file_idx])
        
        # Convert NEF file to PIL image
        img_PIL = NEF2PIL(NEF_path)

        # Resize the PIL image to target size, following used original size as target size
        target_size = img_PIL.size
        resize_PIL(img_PIL, target_size)

        # save PIL image as JPG image in target folder
        save_image_PIL(img_PIL, JPG_folder_path, image_name)
        
    return ("NEF files all saved as JPG at ", JPG_folder_path)


## test the function
NEF_folder_path = "C:/Users/EyesHigh/Desktop/WeddingImageProcessing/data/image1"
JPG_folder_path = "C:/Users/EyesHigh/Desktop/WeddingImageProcessing/data/JPGs"

NEFfolder2JPGfolder(NEF_folder_path, JPG_folder_path)


