# Use the Erin and Dan Wedding images to visualize the data
import os
import rawpy
import pandas as pd
import numpy as np

def NEFImage2Features(NEF_folder_path):

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

def XMP_parameter(xmp_file_path):
    fd = open(xmp_file_path)
    d = fd.read()

    parameter = []

    xmp_start = d.find('crs:Exposure2012=')
    xmp_end = d.find('crs:Contrast2012=')
    xmp_Exposure = d[xmp_start+16:xmp_end-1]
    Exposure = xmp_Exposure.split("\"")[1]
    Exposure = float(Exposure)
    parameter.append(Exposure)


    xmp_start = d.find('crs:Contrast2012=')
    xmp_end = d.find('crs:Highlights2012=')
    xmp_Contrast = d[xmp_start+17:xmp_end-1]
    Contrast = xmp_Contrast.split("\"")[1]
    Contrast = int(Contrast)
    parameter.append(Contrast)
    

    xmp_start = d.find('crs:Highlights2012=')
    xmp_end = d.find('crs:Shadows2012=')
    xmp_Highlight = d[xmp_start+19:xmp_end-1]
    Highlight = xmp_Highlight.split("\"")[1]
    Highlight = int(Highlight)
    parameter.append(Highlight)
    

    xmp_start = d.find('crs:Shadows2012=')
    xmp_end = d.find('crs:Whites2012=')
    xmp_Shadows = d[xmp_start+16:xmp_end-1]
    Shadows = xmp_Shadows.split("\"")[1]
    Shadows = int(Shadows)
    parameter.append(Shadows)
    

    xmp_start = d.find('crs:Temperature=')
    xmp_end = d.find('crs:Tint=')
    xmp_Temperature = d[xmp_start+16:xmp_end-1]
    Temperature = xmp_Temperature.split("\"")[1]
    Temperature = int(Temperature)
    parameter.append(Temperature)

    fd.close()

    return parameter

def XMPFolder2LabelDataFrame(XMP_folder_path):
    import os 
    # go through all XMP files in folder
    files = sorted(os.listdir(XMP_folder_path))
    Edit_labels = []
    Names = []

    for file_idx in range(len(files)):
        file_name = files[file_idx]
        XMP_file_path = os.path.join(XMP_folder_path,file_name)
        parameter = XMP_parameter(XMP_file_path)
        Edit_labels.append(parameter)
        Names.append(file_name)
    
    LabelsDataFrame = pd.DataFrame(Edit_labels)
    LabelsDataFrame.columns = ["Exposure", "Contrast", "Highlights", "Shadows", "Temperature"]
    Labels_df.index = Names

    return LabelsDataFrame

XMP_folder_path = "/Users/user7/Downloads/HD6_12er/Erin & Dan Wedding/ED_DataExploration/ED_DataExploration_XMP" 


