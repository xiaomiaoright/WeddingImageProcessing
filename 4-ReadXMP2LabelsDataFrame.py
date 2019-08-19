# Read in XMP Files and Extract the Exposure, Contrast, Highlight, Shadow, and Temperature parameters
import os 

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
        Names.append(file_name)
    
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

ED_XMP_IndoorPerson = "/Users/user7/Downloads/HD6_12er/Erin & Dan Wedding/ED_XMP_IndoorPerson1"
ED_Label_IndoorPerson = XMPFolder2LabelDataFrame(ED_XMP_IndoorPerson) # There is a problem!
ED_Label_IndoorPerson

JY_Label_IndoorPerson = XMPFolder2LabelDataFrame(JY_XMP_IndoorPerson)
JY_Label_IndoorPerson
JY_Label_IndoorPerson.describe()

LM_Label_IndoorPerson = XMPFolder2LabelDataFrame(LM_XMP_IndoorPerson)
LM_Label_IndoorPerson.describe()

LC_Label_IndoorPerson = XMPFolder2LabelDataFrame(LC_XMP_IndoorPerson)
LC_Label_IndoorPerson.describe()

ED_XMP


