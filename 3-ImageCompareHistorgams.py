# To compare the NEF images of different categories pixel distribution
import os
import rawpy
import imageio
import matplotlib.pyplot as plt
from matplotlib import image as mp_image

def NEF_distribution(NEF_folder_path):

    # go through all NEF files in folder
    NEF_files = sorted(os.listdir(NEF_folder_path))

    # Get the images and show histogram
    for file_idx in range(len(NEF_files)):
        # get the NEF image name
        ImageName = NEF_files[file_idx]

        # get the NEF image file path
        NEF_path = os.path.join(NEF_folder_path, NEF_files[file_idx])

        rp_image = rawpy.imread(NEF_path)
        image = rp_image.postprocess() # rgb image, np.array

        # plot the images, image histogram, and CDF 
        fig = plt.figure(figsize=(12,4))
        
        a = fig.add_subplot(1,3,1)
        imgplot = plt.imshow(image)
        a.set_title(ImageName)

        a = fig.add_subplot(1,3,2)
        imgplot = plt.hist(image.ravel())
        a.set_title("Histogram")

        a = fig.add_subplot(1,3,3)
        imgplot = plt.hist(image.ravel(), bins=255, cumulative=True)
        a.set_title("CDF")
       
        plt.show()

NEF_folder_path = "/Users/user7/Downloads/HD6_12er/Erin & Dan Wedding/ED_DataExploration/ED_DataExploration_NEF"
NEF_distribution(NEF_folder_path)
