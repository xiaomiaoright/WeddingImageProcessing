# to find the image with the smallest dimension size
# Target image size should be smaller than the smallest dimensions of all images
# convert NEF file to PIL image
import numpy as np

def NEF2PIL(nef_image_path):
    import rawpy
    import imageio
    from PIL import Image
    rp_image = rawpy.imread(nef_image_path)
    rgb = rp_image.postprocess()
    rgb_pil = Image.fromarray(rgb)
    return rgb_pil

# Find the dimensions of all images
def FindDimensions(NEF_folder_path):
    import os 

    # Set the list to store all image dimensions
    dimensions = []
    name = []

    # go through all XMP files in folder
    NEF_files = os.listdir(NEF_folder_path)

    
    # Get the images and show the predicted classes
    for file_idx in range(len(NEF_files)):
        
        # set a list to save image name, h, w
        image_size = []

        # save image name
        image = NEF_files[file_idx]
        name.append(image)

        # get the NEF image file path
        NEF_path = os.path.join(NEF_folder_path, NEF_files[file_idx])
        
        # Convert NEF file to PIL image
        img_PIL = NEF2PIL(NEF_path)

        # Get the image size
        h, w = img_PIL.size

        # add image size to image_info
        image_size.append(h)
        image_size.append(w)

        dimensions.append(image_size)
        
    return dimensions, name


NEF_folder_path = "C:/Users/EyesHigh/Desktop/WeddingImageProcessing/data/image1"
file_dimension, file_name = FindDimensions(NEF_folder_path)

h_min = min(file_dimension[0])
w_min = min(file_dimension[1])

print(h_min, w_min)

