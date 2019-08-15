# Read in XMP Files and Extract the Exposure, Contrast, Highlight, Shadow, and Temperature parameters
import os 
import pandas as pd
import numpy as np

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



##--->>> Save IndoorThings Categories
ED_XMP_IndoorThings = "/Users/user7/Downloads/HD6_12er/Erin & Dan Wedding/2017-09-23/ED_XMP_IndoorThings"
JY_XMP_IndoorThings = "/Users/user7/Downloads/HD6_12er/James & Yasmin Wedding/2017-08-06/JY_XMP_IndoorThings"
LM_XMP_IndoorThings = "/Users/user7/Downloads/HD6_12er/Lauren & Matt Wedding/2017-09-08/LM_XMP_IndoorThings"
LC_XMP_IndoorThings = "/Users/user7/Downloads/HD6_12er/Lorraine & Chad Wedding/2017-08-19/LC_XMP_IndoorThings"

# ED folder need to remove the .DS_Store files first
os.remove("/Users/user7/Downloads/HD6_12er/Erin & Dan Wedding/2017-09-23/ED_XMP_IndoorThings/.DS_Store")
os.remove("/Users/user7/Downloads/HD6_12er/James & Yasmin Wedding/2017-08-06/JY_XMP_IndoorThings/.DS_Store")
os.remove("/Users/user7/Downloads/HD6_12er/Lauren & Matt Wedding/2017-09-08/LM_XMP_IndoorThings/.DS_Store")
os.remove("/Users/user7/Downloads/HD6_12er/Lorraine & Chad Wedding/2017-08-19/LC_XMP_IndoorThings/.DS_Store")

# Extract the 5 parameters from XMP and save as dataframe
ED_Label_IndoorThings = XMPFolder2LabelDataFrame(ED_XMP_IndoorThings)
JY_Label_IndoorThings = XMPFolder2LabelDataFrame(JY_XMP_IndoorThings)
LM_Label_IndoorThings = XMPFolder2LabelDataFrame(LM_XMP_IndoorThings)
LC_Label_IndoorThings = XMPFolder2LabelDataFrame(LC_XMP_IndoorThings)

# Save the Dataframe to csv fiels
def DataFrame2CSV(df, csv_folder_path, csv_file_name):
    
    csv_file_path = os.path.join(csv_folder_path, csv_file_name)
    df.to_csv(csv_file_path)
csv_folder_path = "/Users/user7/Downloads/HD6_12er/DataPreparation"
DataFrame2CSV(ED_Label_IndoorThings, csv_folder_path, "ED_Label_IndoorThings.csv")
DataFrame2CSV(JY_Label_IndoorThings, csv_folder_path, "JY_Label_IndoorThings.csv")
DataFrame2CSV(LM_Label_IndoorThings, csv_folder_path, "LM_Label_IndoorThings.csv")
DataFrame2CSV(LC_Label_IndoorThings, csv_folder_path, "LC_Label_IndoorThings.csv")



##--->>> Save OutdoorPerson Categories

ED_XMP_OutdoorPerson = "/Users/user7/Downloads/HD6_12er/Erin & Dan Wedding/2017-09-23/ED_XMP_OutdoorPerson"
JY_XMP_OutdoorPerson = "/Users/user7/Downloads/HD6_12er/James & Yasmin Wedding/2017-08-06/JY_XMP_OutdoorPerson"
LM_XMP_OutdoorPerson = "/Users/user7/Downloads/HD6_12er/Lauren & Matt Wedding/2017-09-08/LM_XMP_OutdoorPerson"
LC_XMP_OutdoorPerson = "/Users/user7/Downloads/HD6_12er/Lorraine & Chad Wedding/2017-08-19/LC_XMP_OutdoorPerson"

# ED folder need to remove the .DS_Store files first
os.remove("/Users/user7/Downloads/HD6_12er/Erin & Dan Wedding/2017-09-23/ED_XMP_OutdoorPerson/.DS_Store")
os.remove("/Users/user7/Downloads/HD6_12er/James & Yasmin Wedding/2017-08-06/JY_XMP_OutdoorPerson/.DS_Store")
os.remove("/Users/user7/Downloads/HD6_12er/Lauren & Matt Wedding/2017-09-08/LM_XMP_OutdoorPerson/.DS_Store")
os.remove("/Users/user7/Downloads/HD6_12er/Lorraine & Chad Wedding/2017-08-19/LC_XMP_OutdoorPerson/.DS_Store")

# Extract the 5 parameters from XMP and save as dataframe
ED_Label_OutdoorPerson = XMPFolder2LabelDataFrame(ED_XMP_OutdoorPerson)
JY_Label_OutdoorPerson = XMPFolder2LabelDataFrame(JY_XMP_OutdoorPerson)
LM_Label_OutdoorPerson = XMPFolder2LabelDataFrame(LM_XMP_OutdoorPerson)
LC_Label_OutdoorPerson = XMPFolder2LabelDataFrame(LC_XMP_OutdoorPerson)

# Save the Dataframe to csv fiels
def DataFrame2CSV(df, csv_folder_path, csv_file_name):
    
    csv_file_path = os.path.join(csv_folder_path, csv_file_name)
    df.to_csv(csv_file_path)
csv_folder_path = "/Users/user7/Downloads/HD6_12er/DataPreparation"
DataFrame2CSV(ED_Label_OutdoorPerson, csv_folder_path, "ED_Label_OutdoorPerson.csv")
DataFrame2CSV(JY_Label_OutdoorPerson, csv_folder_path, "JY_Label_OutdoorPerson.csv")
DataFrame2CSV(LM_Label_OutdoorPerson, csv_folder_path, "LM_Label_OutdoorPerson.csv")
DataFrame2CSV(LC_Label_OutdoorPerson, csv_folder_path, "LC_Label_OutdoorPerson.csv")


##--->>> Save OutdoorThings Categories
ED_XMP_OutdoorThings = "/Users/user7/Downloads/HD6_12er/Erin & Dan Wedding/2017-09-23/ED_XMP_OutdoorThings"
JY_XMP_OutdoorThings = "/Users/user7/Downloads/HD6_12er/James & Yasmin Wedding/2017-08-06/JY_XMP_OutdoorThings"
LM_XMP_OutdoorThings = "/Users/user7/Downloads/HD6_12er/Lauren & Matt Wedding/2017-09-08/LM_XMP_OutdoorThings"
LC_XMP_OutdoorThings = "/Users/user7/Downloads/HD6_12er/Lorraine & Chad Wedding/2017-08-19/LC_XMP_OutdoorThings"

# ED folder need to remove the .DS_Store files first
os.remove("/Users/user7/Downloads/HD6_12er/Erin & Dan Wedding/2017-09-23/ED_XMP_OutdoorThings/.DS_Store")
os.remove("/Users/user7/Downloads/HD6_12er/James & Yasmin Wedding/2017-08-06/JY_XMP_OutdoorThings/.DS_Store")
os.remove("/Users/user7/Downloads/HD6_12er/Lauren & Matt Wedding/2017-09-08/LM_XMP_OutdoorThings/.DS_Store")
os.remove("/Users/user7/Downloads/HD6_12er/Lorraine & Chad Wedding/2017-08-19/LC_XMP_OutdoorThings/.DS_Store")

# Extract the 5 parameters from XMP and save as dataframe
ED_Label_OutdoorThings = XMPFolder2LabelDataFrame(ED_XMP_OutdoorThings)
JY_Label_OutdoorThings = XMPFolder2LabelDataFrame(JY_XMP_OutdoorThings)
LM_Label_OutdoorThings = XMPFolder2LabelDataFrame(LM_XMP_OutdoorThings)
LC_Label_OutdoorThings = XMPFolder2LabelDataFrame(LC_XMP_OutdoorThings)

# Save the Dataframe to csv fiels
def DataFrame2CSV(df, csv_folder_path, csv_file_name):
    
    csv_file_path = os.path.join(csv_folder_path, csv_file_name)
    df.to_csv(csv_file_path)
csv_folder_path = "/Users/user7/Downloads/HD6_12er/DataPreparation"
DataFrame2CSV(ED_Label_OutdoorThings, csv_folder_path, "ED_Label_OutdoorThings.csv")
DataFrame2CSV(JY_Label_OutdoorThings, csv_folder_path, "JY_Label_OutdoorThings.csv")
DataFrame2CSV(LM_Label_OutdoorThings, csv_folder_path, "LM_Label_OutdoorThings.csv")
DataFrame2CSV(LC_Label_OutdoorThings, csv_folder_path, "LC_Label_OutdoorThings.csv")



