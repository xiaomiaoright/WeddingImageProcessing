#### Import Packages
import os
import rawpy
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as nr
import pandas as pd
import rawpy
import scipy.stats as ss
import seaborn as sns

## Plots functions 
def plot_scatter_t(auto_prices, cols, col_y, alpha = 0.1):
    for col in cols:
        fig = plt.figure(figsize=(7,6)) # define plot area
        ax = fig.gca() # define axis   
        auto_prices.plot.scatter(x = col, y = col_y, ax = ax, alpha = alpha)
        ax.set_title('Scatter plot of ' + col_y + ' vs. ' + col) # Give the plot a main title
        ax.set_xlabel(col) # Set text for the x axis
        ax.set_ylabel(col_y)# Set text for y axis
        plt.show()

def plot_desity_2d(auto_prices, cols, col_y = 'price', kind ='kde'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.jointplot(col, col_y, data=auto_prices, kind=kind)
        plt.xlabel(col) # Set text for the x axis
        plt.ylabel(col_y)# Set text for y axis
        plt.show()

# Plot hist_plot of numerical colomns
def hist_plot(vals, lab):
    ## Distribution plot of values
    sns.distplot(vals)
    plt.title('Histogram of ' + lab)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.show()

# plot the histogram of all features and labels
# hist_plot_dataset
def hist_plot_dataset(dataset, col):
    ## Distribution plot of values
    sns.distplot(dataset[col])
    plt.title('Histogram of ' + col)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.show()


csv_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/ED_JY_LC_LM_32bins/ED_JY_LC_LM_Features_Labels_32bins.csv"
DF = pd.read_csv(csv_path, index_col = 0)

DF.head()

DF_ratio = pd.DataFrame()
channel = ["R", "G", "B"]
ratio_cols = []

k = 0 
for i in range(16):
    new_col = str(k) + "/" + str(31-i)
    ratio_cols.append(new_col)
    k = k +1
len(ratio_cols)

col_nums = [0, 32, 64]
old_cols = DF.columns.tolist()
old_cols
k=1
for col_num in col_nums:
    for i in range(16):
        DF_ratio[ratio_cols[(k -1)* 16 + i]] = DF[old_cols[col_num+i]] / DF[old_cols[col_num+(15-i)]]
    k = k +1

DF_ratio.head()

# append the labels to the dataframe
label_cols = ['Exposure',
 'Contrast',
 'Highlights',
 'Shadows',
 'Temperature']

for col in label_cols:
    DF_ratio[col] = DF[col]

DF_ratio.shape
DF_ratio.head()


DF_ratio_new = DF_ratio.drop(labels = "G12/G19", axis = 1)
DF_ratio_new.shape


for col in DF_ration_new.columns:
    hist_plot_dataset(DF_ration_new, col)

cols = DF_ratio.columns.tolist()[:48]
plot_scatter_t(DF_ratio, cols, "Exposure", alpha=0.2)



def NEFFolder2Feature32bins(NEF_folder_path, wedding):

    Features = pd.DataFrame()

    files = sorted(os.listdir(NEF_folder_path))
    

    for file_idx in range(len(files)):
        feature_list = []

        NEF_file_path = os.path.join(NEF_folder_path,files[file_idx])
        

        rp_image = rawpy.imread(NEF_file_path)
        rgb = rp_image.postprocess()

        rgb_sum = rgb.sum(axis = 2).ravel()
        dataset1 = pd.DataFrame({'sum': rgb_sum})

        df_bins16 = dataset1['sum'].value_counts(bins = 32, normalize = True).sort_index()
        col_name = wedding + "_" +files[file_idx].split(".")[0]
        Features[col_name] = df_bins16.tolist()

    return Features.T

from pytictoc import TicToc
t = TicToc() #create instance of class
t.tic()
LM_nef_path = "/Users/user7/Downloads/HD6_12er/Lauren & Matt Wedding/2017-09-08/LM_NEF"
LM_32bins = NEFFolder2Feature32bins(LM_nef_path, "LM")
t.toc()

LM_32bins.head()
LM_32bins.shape

k = 8
col_list = []
for i in range(32):
    col = "rgb" + str(i*k) + "_" + str(i*k+7)
    col_list.append(col)

col_list

LM_32bins.columns = col_list

## join the label columns
LM_labels = pd.read_csv("/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/ED_JY_LC_LM_32bins/LM_Features_Labels_32bins.csv", index_col = 0)
LM_labels.head()
row_names = LM_labels.index.tolist()
row_names
new_row_names = []
for row_name in row_names:
    new_name = "LM_" + row_name
    new_row_names.append(new_name)
new_row_names
LM_labels.index = new_row_names
LM_labels.index.tolist()


LM_32bins_RGB = LM_32bins.join(LM_labels, how = 'inner')
LM_32bins.head()

LM_32bins




# save LM_32bins to csv

def DataFrame2CSV(df, csv_folder_path, csv_file_name):
    
    csv_file_path = os.path.join(csv_folder_path, csv_file_name)
    df.to_csv(csv_file_path)
csv_folder_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDataPrepBins32_RGB"
csv_file_name = "LM_32bins_features.csv"
DataFrame2CSV(LM_32bins, csv_folder_path, csv_file_name)



