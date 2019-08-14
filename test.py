NEF_folder_path = "/Users/user7/Desktop/WeddingImageProcessing/data/image1"
NEF_files = sorted(os.listdir(NEF_folder_path))

Features = pd.DataFrame()

for file_idx in range(len(NEF_files)):
    # get the iamge name
    image_name = NEF_files[file_idx]
    print(image_name)
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

    dataset = pd.DataFrame({'Red': rgb_1, 'Green': rgb_2, 'Blue':rgb_3})

    df_1 = dataset['Red'].value_counts(bins = 16, normalize = True).sort_index()
    df_2 = dataset['Green'].value_counts(bins = 16, normalize = True).sort_index()
    df_3 = dataset['Blue'].value_counts(bins = 16, normalize = True).sort_index()

    df_new = df_1.append(df_2)
    df_new_new = df_new.append(df_3)
    print(df_new_new)

    temp_df = pd.DataFrame({image_name: df_new_new})
    print(temp_df)

    Features[image_name] = temp_df[image_name]
    #print(Features)
Features[image_name] = temp_df[image_name]
Features


def NEF2Feature(NEF_files_list, NEF_files_name):

    temp_list = []
    i = 0 
    Features = pd.DataFrame()

    for file_path in NEF_files_list:
        rp_image = rawpy.imread(NEF_path)   
        rgb = rp_image.postprocess()
        rgb_1 = np.sort(rgb[:,:,0].ravel())
        rgb_2 = np.sort(rgb[:,:,1].ravel())
        rgb_3 = np.sort(rgb[:,:,2].ravel())

        dataset = pd.DataFrame({'Red': rgb_1, 'Green': rgb_2, 'Blue':rgb_3})

        df_name = pd.DataFrame([NEF_files_name[i]], columns = ["fileName"])
        df_0 = df_name['fileName']
        temp_list.append(df_0.tolist())

        df_1 = dataset['Red'].value_counts(bins = 16, normalize = True).sort_index()
        temp_list.append(df_1.tolist())


        df_2 = dataset['Green'].value_counts(bins = 16, normalize = True).sort_index()
        temp_list.append(df_2.tolist())


        df_3 = dataset['Blue'].value_counts(bins = 16, normalize = True).sort_index()
        temp_list.append(df_3.tolist())

        t_list = np.array(temp_list).ravel() 

        print(t_list)
        #Features[str(NEF_files_name[i])] = t_list
    


NEF_files_list = ['/Users/user7/Desktop/WeddingImageProcessing/data/image1/baby.nef','/Users/user7/Desktop/WeddingImageProcessing/data/image1/infrared.nef']
NEF_files_name = ['baby', 'infrared']
NEF2Feature(NEF_files_list, NEF_files_name)


data = {'Name': ['Jai', 'Princi', 'Gaurav', 'Anuj'], 
        'Height': [5.1, 6.2, 5.1, 5.2], 
        'Qualification': ['Msc', 'MA', 'Msc', 'Msc']} 
  
# Convert the dictionary into DataFrame 
df = pd.DataFrame(data) 
df
Features = pd.DataFrame()
for name in ["Name", "Height", "Qualification"]:
    Features[name] = df[name]

Features

NEF_path = "/Users/user7/Desktop/WeddingImageProcessing/data/image1/baby.nef"
    

def NEFPath2FeatureDF(NEF_path):
    
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


    df_new = df_1.append(df_2)
    df_new_new = df_new.append(df_3)

    # save R, G, B channel bins count into one column of a new temporary dataframe
    temp_df = pd.DataFrame({"rgb": df_new_new})

    Features = pd.DataFrame()
    Features['rgb'] = temp_df['rgb']
