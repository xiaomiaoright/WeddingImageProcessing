# Count the number of files given a folder path

def CountFiles(path):
    import os 
    # go through all XMP files in folder
    files = sorted(os.listdir(path))
    count = 0 
    file_name_list = []
    # Get the images and show the predicted classes
    for file_idx in range(len(files)):
        # set the JPG iamge name
        count = count + 1
        file_name = files[file_idx]
        file_name_list.append(file_name)
        
    return count, file_name_list

NEF_file_path = "/Users/user7/Downloads/HD6_12er/Erin & Dan Wedding/2017-09-23/ED_NEF"
count, file_name = CountFiles(NEF_file_path)
print("The number of NEF files", count)
print(file_name[:10])

XPM_file_path = "/Users/user7/Downloads/HD6_12er/Erin & Dan Wedding/2017-09-23/ED_XMP_IndoorThings"
XMP_count, XMP_file_name = CountFiles(XPM_file_path)
print("The number of XMP files", XMP_count) 

#ED_XMP 878
#ED_XMP_IndoorPerson 350
#ED_XMP_IndoorThings 66
#ED_XMP_OutdoorPerson
#ED_XMP_OutdoorThings
print(XMP_file_name[:10])

