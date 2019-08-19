## Step1: Extract all Features DataFrame
## Step2: Extract all Labels DataFrame
## Step3: Explore data by visulization
## Step4: Build model

##-->>> progress update:
##-->>> ED wedding Feature Dataframe is saved to csv.
##-->>> JY, LM, LC wedding Features DataFrames is in progress


#### Import Packages
import os
import rawpy
import pandas as pd
import numpy as np



#### Get the Feature Dataframe from NEF files
# Define the NEF2DF function
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

# Set the NEF files folder, the folder has all the NEF files for the wedding
ED_NEF = "/Users/user7/Downloads/HD6_12er/Erin & Dan Wedding/2017-09-23/ED_NEF"
JY_NEF = "/Users/user7/Downloads/HD6_12er/James & Yasmin Wedding/2017-08-06/JY_NEF"
LM_NEF = "/Users/user7/Downloads/HD6_12er/Lauren & Matt Wedding/2017-09-08/LM_NEF"
LC_NEF = "/Users/user7/Downloads/HD6_12er/Lorraine & Chad Wedding/2017-08-19/LC_NEF"

# Remove the .DS_Store file in ED_NEF folder
os.remove("/Users/user7/Downloads/HD6_12er/Erin & Dan Wedding/2017-09-23/ED_NEF/.DS_Store")
os.remove("/Users/user7/Downloads/HD6_12er/James & Yasmin Wedding/2017-08-06/JY_NEF/.DS_Store")
os.remove("/Users/user7/Downloads/HD6_12er/Lauren & Matt Wedding/2017-09-08/LM_NEF/.DS_Store")
os.remove("/Users/user7/Downloads/HD6_12er/Lorraine & Chad Wedding/2017-08-19/LC_NEF/.DS_Store")

# Get the Feature DataFrame for four weddings seperately
ED_Feature_Df = NEFFolder2Feature(ED_NEF)
JY_Feature_Df = NEFFolder2Feature(JY_NEF)
LM_Feature_Df = NEFFolder2Feature(LM_NEF)
LC_Feature_Df = NEFFolder2Feature(LC_NEF)

# Save the Feature DataFrame for future usage
def DataFrame2CSV(df, csv_folder_path, csv_file_name):
    
    csv_file_path = os.path.join(csv_folder_path, csv_file_name)
    df.to_csv(csv_file_path)
csv_folder_path = "/Users/user7/Downloads/HD6_12er/DataPreparation"

DataFrame2CSV(ED_Feature_Df, csv_folder_path, "ED_Feature.csv")
DataFrame2CSV(JY_Feature_Df, csv_folder_path, "JY_Feature.csv")
DataFrame2CSV(LM_Feature_Df, csv_folder_path, "LM_Feature.csv")
DataFrame2CSV(LC_Feature_Df, csv_folder_path, "LC_Feature.csv")


##### Get the Label DataFrame from the XMP files for the IndoorPerson category only
# Define the function of XMP2LableDF
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

#f = 'C:/Users/EyesHigh/Desktop/WeddingImageProcessing/data/XMPs/800_1743-1.xmp'
#print(XMP_parameter(f))


def XMPFolder2LabelList(XMP_folder_path):

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

    return Edit_labels, Names
    
def XMPFolder2LabelDataFrame(XMP_folder_path):

    # go through all XMP files in folder
    files = sorted(os.listdir(XMP_folder_path))
    Edit_labels = []
    Names = []

    for file_idx in range(len(files)):
        file_name = files[file_idx]
        XMP_file_path = os.path.join(XMP_folder_path,file_name)
        parameter = XMP_parameter(XMP_file_path)
        Edit_labels.append(parameter)
        Names.append(file_name.split(".")[0])
    
    LabelsDataFrame = pd.DataFrame(Edit_labels)
    LabelsDataFrame.columns = ["Exposure", "Contrast", "Highlights", "Shadows", "Temperature"]
    LabelsDataFrame.index = Names

    return LabelsDataFrame

ED_XMP_IndoorPerson = "/Users/user7/Downloads/HD6_12er/Erin & Dan Wedding/2017-09-23/ED_XMP_IndoorPerson"
JY_XMP_IndoorPerson = "/Users/user7/Downloads/HD6_12er/James & Yasmin Wedding/2017-08-06/JY_XMP_IndoorPerson"
LM_XMP_IndoorPerson = "/Users/user7/Downloads/HD6_12er/Lauren & Matt Wedding/2017-09-08/LM_XMP_IndoorPerson"
LC_XMP_IndoorPerson = "/Users/user7/Downloads/HD6_12er/Lorraine & Chad Wedding/2017-08-19/LC_XMP_IndoorPerson"

# ED folder need to remove the .DS_Store files first
os.remove("/Users/user7/Downloads/HD6_12er/Erin & Dan Wedding/2017-09-23/ED_XMP_IndoorPerson/.DS_Store")

ED_Label_IndoorPerson_Df = XMPFolder2LabelDataFrame(ED_XMP_IndoorPerson) 
JY_Label_IndoorPerson_Df = XMPFolder2LabelDataFrame(JY_XMP_IndoorPerson)
LM_Label_IndoorPerson_Df = XMPFolder2LabelDataFrame(LM_XMP_IndoorPerson)
LC_Label_IndoorPerson_Df = XMPFolder2LabelDataFrame(LC_XMP_IndoorPerson)

# Save the Label DataFrame for future usage
csv_folder_path = "/Users/user7/Downloads/HD6_12er/DataPreparation"

DataFrame2CSV(ED_Label_IndoorPerson_Df, csv_folder_path, "ED_Label_IndoorPerson.csv")
DataFrame2CSV(JY_Label_IndoorPerson_Df, csv_folder_path, "JY_Label_IndoorPerson.csv")
DataFrame2CSV(LM_Label_IndoorPerson_Df, csv_folder_path, "LM_Label_IndoorPerson.csv")
DataFrame2CSV(LC_Label_IndoorPerson_Df, csv_folder_path, "LC_Label_IndoorPerson.csv")


## Merge two DataFrame using .concat
ED_Df = ED_Feature_Df.join(ED_Label_IndoorPerson_Df, how='inner')
JY_Df = JY_Feature_Df.join(JY_Label_IndoorPerson_Df, how='inner')
LM_Df = LM_Feature_Df.join(LM_Label_IndoorPerson_Df, how='inner')
LC_Df = LC_Feature_Df.join(LC_Label_IndoorPerson_Df, how='inner')

ED_Feature_Df.describe()

JY_Feature_Df



def NEFFolder2Feature32bins(NEF_folder_path):
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

        df_1 = dataset['Red'].value_counts(bins = 32, normalize = True).sort_index()
        df_2 = dataset['Green'].value_counts(bins = 32, normalize = True).sort_index()
        df_3 = dataset['Blue'].value_counts(bins = 32, normalize = True).sort_index()

        feature_list.append(df_1.tolist())
        feature_list.append(df_2.tolist())
        feature_list.append(df_3.tolist())

        feature_array = np.array(feature_list).ravel()

        Features[files[file_idx].split(".")[0]] = feature_array

    return Features.T

from pytictoc import TicToc
t = TicToc() #create instance of class
t.tic()
ED_Feature_32bins = NEFFolder2Feature32bins(ED_NEF)
t.toc()

ED_Feature_32bins.head()


# Rename the columns names
channels = ['R', 'G', 'B']
new_columns = []
for ch in channels:
    k = 0
    for i in range(32):
        col_name = ch + str(k) + "_" + str(k+7)
        new_columns.append(col_name)
        k = k+8
ED_Feature_32bins.columns = new_columns
ED_Feature_32bins.head()

csv_folder_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32"

DataFrame2CSV(ED_Feature_32bins, csv_folder_path, "ED_Features_32bins.csv")


## Append the features
ED_IP_Label_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDataPrepBins16/ED_Label_IndoorPerson.csv"
ED_IP_Labels = pd.read_csv(ED_IP_Label_path, index_col = 0)

ED_IP_Labels.head()

ED_IP_Features_Labels_32bins = ED_Feature_32bins.join(ED_IP_Labels, how = "inner")

ED_IP_Features_Labels_32bins.head()

def DataFrame2CSV(df, csv_folder_path, csv_file_name):
    
    csv_file_path = os.path.join(csv_folder_path, csv_file_name)
    df.to_csv(csv_file_path)

csv_folder_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32"
csv_file_name_IP = "ED_IP_Features_Labels_32bins.csv"
DataFrame2CSV(ED_IP_Features_Labels_32bins, csv_folder_path, csv_file_name_IP)
ED_Feature_32bins.shape
ED_IP_Features_Labels_32bins.shape