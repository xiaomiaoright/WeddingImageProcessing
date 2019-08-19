# Read through NEF folder 
# Extract and normalize the distribution of R, G, B pixel values in bins of 16 as image feature
# Save the features of all images in the folder to one Features dataframe
# Two methods are created to read in the NEF file path and create Feature dataframe
import os
import rawpy
import pandas as pd
import numpy as np
from pytictoc import TicToc


#### Method 1: Combine the NEFFolder2FeatureList function and NEFfile2FeatureDF function
def NEFFolder2FeatureList(NEF_folder_path):
    import os 
    # go through all XMP files in folder
    files = sorted(os.listdir(NEF_folder_path))
    Features = []

    for file_idx in range(len(files)):
        feature_list = []

        file_name = files[file_idx]
        NEF_file_path = os.path.join(NEF_folder_path,file_name)

        rp_image = rawpy.imread(NEF_file_path)
        rgb = rp_image.postprocess()

        # red the R, G, B channels seperately and sort from 0 to 255
        rgb_1 = np.sort(rgb[:,:,0].ravel())
        rgb_2 = np.sort(rgb[:,:,1].ravel())
        rgb_3 = np.sort(rgb[:,:,2].ravel())

        dataset = pd.DataFrame({'Red': rgb_1, 'Green': rgb_2, 'Blue':rgb_3})

        df_1 = dataset['Red'].value_counts(bins = 16, normalize = True).sort_index()
        df_2 = dataset['Green'].value_counts(bins = 16, normalize = True).sort_index()
        df_3 = dataset['Blue'].value_counts(bins = 16, normalize = True).sort_index()

        feature_list.append(df_1.tolist())
        feature_list.append(df_2.tolist())
        feature_list.append(df_3.tolist())

        Features.append(feature_list)

    return Features

def NEFfile2FeatureDF(NEF_folder_path):
    FeatureList = NEFFolder2FeatureList(NEF_folder_path)
    files = sorted(os.listdir(NEF_folder_path))

    FeatureArray = np.array(FeatureList)
    Features = pd.DataFrame()

    for i in range(FeatureArray.shape[0]):
        image_array = FeatureArray[i,:,:].ravel()
        Features[files[i]] = image_array
    
    return Features.T

t = TicToc() #create instance of class
t.tic()
NEF_folder_path = "/Users/user7/Desktop/WeddingImageProcessing/data/NEFs_test"
Features_DF = NEFfile2FeatureDF(NEF_folder_path)
Features_DF # no errow message!!!!
t.toc() ## 114s



##### Method 2: NEFFolder2Feature function
def NEFFolder2Feature(NEF_folder_path):
    import os 

    Features = pd.DataFrame()

    # go through all XMP files in folder
    files = sorted(os.listdir(NEF_folder_path))
    

    for file_idx in range(len(files)):
        feature_list = []

        NEF_file_path = os.path.join(NEF_folder_path,files[file_idx])

        rp_image = rawpy.imread(NEF_file_path)
        rgb = rp_image.postprocess()

        # red the R, G, B channels seperately and sort from 0 to 255
        rgb_1 = np.sort(rgb[:,:,0].ravel())
        rgb_2 = np.sort(rgb[:,:,1].ravel())
        rgb_3 = np.sort(rgb[:,:,2].ravel())

        dataset = pd.DataFrame({'Red': rgb_1, 'Green': rgb_2, 'Blue':rgb_3})

        df_1 = dataset['Red'].value_counts(bins = 16, normalize = True).sort_index()
        df_2 = dataset['Green'].value_counts(bins = 16, normalize = True).sort_index()
        df_3 = dataset['Blue'].value_counts(bins = 16, normalize = True).sort_index()

        feature_list.append(df_1.tolist())
        feature_list.append(df_2.tolist())
        feature_list.append(df_3.tolist())

        feature_array = np.array(feature_list).ravel()

        Features[files[file_idx].split(".")[0]] = feature_array

    return Features.T

t = TicToc()
t.tic()
NEF_folder_path = "/Users/user7/Desktop/WeddingImageProcessing/data/NEFs_test"
Features_DF2 = NEFFolder2Feature(NEF_folder_path)
Features_DF2
t.toc() ### 112s


## to save dataframe to csv files
def DataFrame2CSV(df, csv_folder_path, csv_file_name):
    
    csv_file_path = os.path.join(csv_folder_path, csv_file_name)
    df.to_csv(csv_file_path)
csv_folder_path = "/Users/user7/Desktop/WeddingImageProcessing/data"
csv_file_name = "Features_DF.csv"

DataFrame2CSV(Features_DF, csv_folder_path, csv_file_name)






