# Read in XMP Files and Extract the Exposure, Contrast, Highlight, Shadow, and Temperature parameters
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

    return Edit_labels, Names

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
    LabelsDataFrame.index = Names

    return LabelsDataFrame

XMP_file_path = '/Users/user7/Downloads/HD6_12er/Erin & Dan Wedding/ED_DataExploration/ED_DataExploration_XMP'
Df = XMPFolder2LabelDataFrame(XMP_file_path)
Df
Labels, Names = XMPFolder2LabelList(XMP_file_path)
print(Labels)
print(Names)
print(len(Labels))
Labels = np.array(Labels)
Labels.shape

Labels_df = pd.DataFrame(Labels)
Labels_df.columns=["Exposure", "Contrast", "Highlights", "Shadows", "Temperature"]
Labels_df.index = Names


ED_XMP_IndoorPerson = "/Users/user7/Downloads/HD6_12er/Erin & Dan Wedding/2017-09-23/ED_XMP_IndoorPerson"
JY_XMP_IndoorPerson = "/Users/user7/Downloads/HD6_12er/James & Yasmin Wedding/2017-08-06/JY_XMP_IndoorPerson"
LM_XMP_IndoorPerson = "/Users/user7/Downloads/HD6_12er/Lauren & Matt Wedding/2017-09-08/LM_XMP_IndoorPerson"
LC_XMP_IndoorPerson = "/Users/user7/Downloads/HD6_12er/Lorraine & Chad Wedding/2017-08-19/LC_XMP_IndoorPerson"


ED_Label_IndoorPerson = XMPFolder2LabelDataFrame(ED_XMP_IndoorPerson) # There is a problem!

JY_Label_IndoorPerson = XMPFolder2LabelDataFrame(JY_XMP_IndoorPerson)
JY_Label_IndoorPerson.describe()

LM_Label_IndoorPerson = XMPFolder2LabelDataFrame(LM_XMP_IndoorPerson)
LM_Label_IndoorPerson.describe()

LC_Label_IndoorPerson = XMPFolder2LabelDataFrame(LC_XMP_IndoorPerson)
LC_Label_IndoorPerson.describe()



import os
### Test the XMP IndoorPerson folders
# !!!!!! There is a problem with ED IndoorPerson folder, need to update
XMP_folder_path = "/Users/user7/Downloads/HD6_12er/Erin & Dan Wedding/2017-09-23/ED_NEF"
files = sorted(os.listdir(XMP_folder_path))
Edit_labels = []
Names = []

for file_idx in range(len(files)):
    file_name = files[file_idx]
    XMP_file_path = os.path.join(XMP_folder_path,file_name)
    parameter = XMP_parameter(XMP_file_path)
    print(file_name)
    print(parameter)
    
    Edit_labels.append(parameter)
    Names.append(file_name)



# JY XMP IndoorPerson folder has no problem
XMP_folder_path = "/Users/user7/Downloads/HD6_12er/James & Yasmin Wedding/2017-08-06/JY_XMP_IndoorPerson"
files = sorted(os.listdir(XMP_folder_path))
Edit_labels = []
Names = []

for file_idx in range(len(files)):
    file_name = files[file_idx]
    XMP_file_path = os.path.join(XMP_folder_path,file_name)
    parameter = XMP_parameter(XMP_file_path)
    print(file_name)
    print(parameter)
    
    Edit_labels.append(parameter)
    Names.append(file_name)

# LM XMP IndoorPerson folder has no problem
LM_XMP_IndoorPerson_folder_path = "/Users/user7/Downloads/HD6_12er/Lauren & Matt Wedding/2017-09-08/LM_XMP_IndoorPerson"
files = sorted(os.listdir(LM_XMP_IndoorPerson_folder_path))
Edit_labels = []
Names = []

for file_idx in range(len(files)):
    file_name = files[file_idx]
    XMP_file_path = os.path.join(LM_XMP_IndoorPerson_folder_path,file_name)
    parameter = XMP_parameter(XMP_file_path)
    print(file_name)
    print(parameter)
    
    Edit_labels.append(parameter)
    Names.append(file_name)

# LC XMP IndoorPerson folder has no problem
LC_XMP_IndoorPerson_folder_path = "/Users/user7/Downloads/HD6_12er/Lorraine & Chad Wedding/2017-08-19/LC_XMP_IndoorPerson"
files = sorted(os.listdir(LC_XMP_IndoorPerson_folder_path))
Edit_labels = []
Names = []

for file_idx in range(len(files)):
    file_name = files[file_idx]
    XMP_file_path = os.path.join(LC_XMP_IndoorPerson_folder_path,file_name)
    parameter = XMP_parameter(XMP_file_path)
    print(file_name)
    print(parameter)
    
    Edit_labels.append(parameter)
    Names.append(file_name)

