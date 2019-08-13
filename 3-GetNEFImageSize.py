# Go throught a NEF images folder
# Get all the NEF images size and save in a list
import rawpy
import imageio
import os

def NEFfolder2ImageSize(NEF_folder_path):

    ImageSize = []
    # go through all XMP files in folder
    NEF_files = sorted(os.listdir(NEF_folder_path))

    # Convert NEF to RGB and save the rgb array size to list
    
    #for file_idx in range(len(NEF_files)): # test with all the images
    for file_idx in range(50): # test with the first 50 images
        # get the NEF image file path
        NEF_path = os.path.join(NEF_folder_path, NEF_files[file_idx])
        
        rp_image = rawpy.imread(NEF_path)
        rgb = rp_image.postprocess()

        ImageSize.append(rgb.shape)

        
    return ImageSize

## test the function with Erin & Dan Wedding folder
NEF_folder_path = "/Users/user7/Downloads/HD6_12er/Erin & Dan Wedding/2017-09-23/ED_NEF"
#NEF_folder_path = "/Users/user7/Desktop/WeddingImageProcessing/data/image1"
imageSize = NEFfolder2ImageSize(NEF_folder_path)

print(imageSize)

## Results showed NEF images are in different sizes. 