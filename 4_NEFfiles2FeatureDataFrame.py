# Read through NEF folder 
# Extract and normalize the distribution of R, G, B pixel values in bins of 16 as image feature
# Save the features of all images in the folder to one Features dataframe

import os
import rawpy
import pandas as pd
import numpy as np

def NEFImage2Features(NEF_folder_path)):

    # create a feature dataframe
    Features = pd.DataFrame()
    
    # go through all NEF files in folder
    NEF_files = sorted(os.listdir(NEF_folder_path))
    
    # Get the images and show the predicted classes
    for file_idx in range(len(NEF_files)):
        # get the iamge name
        image_name = NEF_files[file_idx]

        # get the NEF image file path
        NEF_path = os.path.join(NEF_folder_path, NEF_files[file_idx])

        # read the NEF image as RGB
        import rawpy
        import imageio
    
        rp_image = rawpy.imread(NEF_path)
        rgb = rp_image.postprocess()

        # red the R, G, B channels seperately and sort from 0 to 255
        rgb_1 = np.sort(rgb[:,:,0].ravel())
        rgb_2 = np.sort(rgb[:,:,1].ravel())
        rgb_3 = np.sort(rgb[:,:,2].ravel())

        # save the R, G, B values to three columns in dataframe
        dataset = pd.DataFrame({'Red': rgb_1, 'Green': rgb_2, 'Blue':rgb_3})

        # value_count the R, G, B channels into 16 bins
        df_1 = dataset['Red'].value_counts(bins = 16, normalize = True).sort_index()
        df_2 = dataset['Green'].value_counts(bins = 16, normalize = True).sort_index()
        df_3 = dataset['Blue'].value_counts(bins = 16, normalize = True).sort_index()

        # save R, G, B channel bins count into one column of a new temporary dataframe
        temp_df = pd.DataFrame({"rgb": df_rgb})
        Features[image_name] = temp_df

    # transpose the Features dataframe to make sure each column represents the a feature
    Features = Features.T

    return Features

    