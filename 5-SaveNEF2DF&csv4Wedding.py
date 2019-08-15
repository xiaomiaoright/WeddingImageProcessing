import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
## Load the data
ED_csv_path = "/Users/user7/Downloads/HD6_12er/DataPreparation/ED_Feature.csv"

# read the file as csv and use the first row as index
ED_Features = pd.read_csv(ED_csv_path, index_col = 0) 
ED_Features.columns.tolist()

# Set the Features Column names
Features_Columns = []
Columns_color = ['R', 'G', 'B']
Columns_list = []

for color in Columns_color:
    k = 0
    for i in range(16):
        col_name = color + str(k) + "-" + str(k+15)
        Features_Columns.append(col_name)
        k = k +16
Features_Columns    

# Set Features column nane
ED_Features.columns = Features_Columns
ED_Features.columns

# Read in the Label_IndoorPerson dataframe
ED_csv_label = "/Users/user7/Downloads/HD6_12er/DataPreparation/ED_Label_IndoorPerson.csv"
ED_Label_IP = pd.read_csv(ED_csv_label, index_col = 0)
ED_Label_IP

## Merge two Dataframe to one
ED_IP_Df = ED_Features.join(ED_Label_IP, how='inner')
ED_IP_Df  ## 236 X53 

## Save the full ED_IP_Df to one csv file

def DataFrame2CSV(df, csv_folder_path, csv_file_name):
    
    csv_file_path = os.path.join(csv_folder_path, csv_file_name)
    df.to_csv(csv_file_path)
ED_csv_folder_path = "/Users/user7/Downloads/HD6_12er/DataPreparation"
ED_data_name = "ED_IP_Features_Labels.csv"
DataFrame2CSV(ED_IP_Df, ED_csv_folder_path, ED_data_name)


#JY dataset

JY_csv_path = "/Users/user7/Downloads/HD6_12er/DataPreparation/JY_Feature.csv"
JY_csv_label = "/Users/user7/Downloads/HD6_12er/DataPreparation/JY_Label_IndoorPerson.csv"

# read the file as csv and use the first row as index
JY_Features = pd.read_csv(JY_csv_path, index_col = 0) 
JY_Label_IP = pd.read_csv(JY_csv_label, index_col = 0)

# Rename of Feature columns
JY_Features.columns = Features_Columns


## Merge two Dataframe to one
JY_IP_Df = JY_Features.join(JY_Label_IP, how='inner')

## Save to data files
JY_csv_folder_path = "/Users/user7/Downloads/HD6_12er/DataPreparation"
JY_data_name = "JY_IP_Features_Labels.csv"
DataFrame2CSV(JY_IP_Df, JY_csv_folder_path, JY_data_name)


## LM Dataset

LM_csv_path = "/Users/user7/Downloads/HD6_12er/DataPreparation/LM_Feature.csv"
LM_csv_label = "/Users/user7/Downloads/HD6_12er/DataPreparation/LM_Label_IndoorPerson.csv"

# read the file as csv and use the first row as index
LM_Features = pd.read_csv(LM_csv_path, index_col = 0) 
LM_Label_IP = pd.read_csv(LM_csv_label, index_col = 0)

# Rename of Feature columns
LM_Features.columns = Features_Columns


## Merge two Dataframe to one
LM_IP_Df = LM_Features.join(LM_Label_IP, how='inner')

## Save to data files
LM_csv_folder_path = "/Users/user7/Downloads/HD6_12er/DataPreparation"
LM_data_name = "LM_IP_Features_Labels.csv"
DataFrame2CSV(LM_IP_Df, LM_csv_folder_path, LM_data_name)



## LC Dataset
LC_csv_path = "/Users/user7/Downloads/HD6_12er/DataPreparation/LC_Feature.csv" 
LC_csv_label = "/Users/user7/Downloads/HD6_12er/DataPreparation/LC_Label_IndoorPerson.csv" 
 
 
# read the file as csv and use the first row as index 
LC_Features = pd.read_csv(LC_csv_path, index_col = 0)  
LC_Label_IP = pd.read_csv(LC_csv_label, index_col = 0) 
 
 
# Rename of Feature columns 
LC_Features.columns = Features_Columns 
 
## Merge two Dataframe to one 
LC_IP_Df = LC_Features.join(LC_Label_IP, how='inner') 
 
 
## Save to data files 
LC_csv_folder_path = "/Users/user7/Downloads/HD6_12er/DataPreparation" 
LC_data_name = "LC_IP_Features_Labels.csv" 
DataFrame2CSV(LC_IP_Df, LC_csv_folder_path, LC_data_name) 
 


## Append ED_IP, JY_IP, LM_IP, LC_IP three dataFrames into one full dataset

ED_df = ED_IP_Df
ED_df.shape
ED_JY_df = ED_df.append(JY_IP_Df)
ED_JY_df.shape
ED_JY_LM_df = ED_JY_df.append(LM_IP_Df)
ED_JY_LM_df.shape ## 993
ED_JY_LM_LC_Df = ED_JY_LM_df.append(LC_IP_Df)
ED_JY_LM_LC_Df.shape # 1609



## save the full ED_JY_LM_df to csv
ED_JY_LM_LC_IP_path = "/Users/user7/Downloads/HD6_12er/DataPreparation"
ED_JY_LM_LC_IP_name = "ED_JY_LM_LC_IP_Features_Labels.csv"
DataFrame2CSV(ED_JY_LM_LC_Df, ED_JY_LM_LC_IP_path, ED_JY_LM_LC_IP_name)





### Work with OutdoorPerson Categories
# the Feature files are the same, only need to read in the label file

# ED_OutdoorPerson 
# read the file as csv and use the first row as index
ED_OP_csv_label = "/Users/user7/Downloads/HD6_12er/DataPreparation/ED_Label_OutdoorPerson.csv"
ED_OP_Label = pd.read_csv(ED_OP_csv_label, index_col = 0)

## Merge two Dataframe to one
ED_OP_Df = ED_Features.join(ED_OP_Label, how='inner')

## Save to data files
ED_OP_csv_folder_path = "/Users/user7/Downloads/HD6_12er/DataPreparation"
ED_OP_data_name = "ED_OP_Features_Labels.csv"
DataFrame2CSV(ED_OP_Df, ED_OP_csv_folder_path, ED_OP_data_name)

## JY_OutdoorPerson
JY_OP_csv_label = "/Users/user7/Downloads/HD6_12er/DataPreparation/JY_Label_OutdoorPerson.csv" 
JY_OP_Label = pd.read_csv(JY_OP_csv_label, index_col = 0) 
 
 
## Merge two Dataframe to one 
JY_OP_Df = JY_Features.join(JY_OP_Label, how='inner') 
 
 
## Save to data files 
JY_OP_csv_folder_path = "/Users/user7/Downloads/HD6_12er/DataPreparation" 
JY_OP_data_name = "JY_OP_Features_Labels.csv" 
DataFrame2CSV(JY_OP_Df, JY_OP_csv_folder_path, JY_OP_data_name) 



## LM_OutdoorPerson
LM_OP_csv_label = "/Users/user7/Downloads/HD6_12er/DataPreparation/LM_Label_OutdoorPerson.csv" 
LM_OP_Label = pd.read_csv(LM_OP_csv_label, index_col = 0) 
 
 
## Merge two Dataframe to one 
LM_OP_Df = LM_Features.join(LM_OP_Label, how='inner') 
 
 
## Save to data files 
LM_OP_csv_folder_path = "/Users/user7/Downloads/HD6_12er/DataPreparation" 
LM_OP_data_name = "LM_OP_Features_Labels.csv" 
DataFrame2CSV(LM_OP_Df, LM_OP_csv_folder_path, LM_OP_data_name) 


## LC_OutdoorPerson
LC_OP_csv_label = "/Users/user7/Downloads/HD6_12er/DataPreparation/LC_Label_OutdoorPerson.csv"  
LC_OP_Label = pd.read_csv(LC_OP_csv_label, index_col = 0)  
## Merge two Dataframe to one  
LC_OP_Df = LC_Features.join(LC_OP_Label, how='inner')  
## Save to data files  
LC_OP_csv_folder_path = "/Users/user7/Downloads/HD6_12er/DataPreparation"  
LC_OP_data_name = "LC_OP_Features_Labels.csv"  
DataFrame2CSV(LC_OP_Df, LC_OP_csv_folder_path, LC_OP_data_name) 



## Append ED_IP, JY_IP, LM_IP three dataFrames into one full dataset

ED_OP = ED_OP_Df
ED_OP.shape
ED_JY_OP = ED_OP.append(JY_OP_Df)
ED_JY_OP.shape
ED_JY_LM_OP = ED_JY_OP.append(LM_OP_Df)
ED_JY_LM_OP.shape ## 612
ED_JY_LM_LC_OP = ED_JY_LM_OP.append(LC_OP_Df)
ED_JY_LM_LC_OP.shape #854


## save the full ED_JY_LM_df to csv
ED_JY_LM_LC_OP_path = "/Users/user7/Downloads/HD6_12er/DataPreparation"
ED_JY_LM_LC_OP_name = "ED_JY_LM_LC_OP_Features_Labels.csv"
DataFrame2CSV(ED_JY_LM_LC_OP, ED_JY_LM_LC_OP_path, ED_JY_LM_LC_OP_name)



## Work with IndoorThings

#ED_IndoorThings
ED_IT_csv_label = "/Users/user7/Downloads/HD6_12er/DataPreparation/ED_Label_IndoorThings.csv"  
ED_IT_Label = pd.read_csv(ED_IT_csv_label, index_col = 0)  
## Merge two Dataframe to one  
ED_IT_Df = ED_Features.join(ED_IT_Label, how='inner')  
## Save to data files  
ED_IT_csv_folder_path = "/Users/user7/Downloads/HD6_12er/DataPreparation"  
ED_IT_data_name = "ED_IT_Features_Labels.csv"  
DataFrame2CSV(ED_IT_Df, ED_IT_csv_folder_path, ED_IT_data_name) 


#JY_IndoorThings

JY_IT_csv_label = "/Users/user7/Downloads/HD6_12er/DataPreparation/JY_Label_IndoorThings.csv"  
JY_IT_Label = pd.read_csv(JY_IT_csv_label, index_col = 0)  
## Merge two Dataframe to one  
JY_IT_Df = JY_Features.join(JY_IT_Label, how='inner')  
## Save to data files  
JY_IT_csv_folder_path = "/Users/user7/Downloads/HD6_12er/DataPreparation"  
JY_IT_data_name = "JY_IT_Features_Labels.csv"  
DataFrame2CSV(JY_IT_Df, JY_IT_csv_folder_path, JY_IT_data_name) 

#LM_IndoorThings
LM_IT_csv_label = "/Users/user7/Downloads/HD6_12er/DataPreparation/LM_Label_IndoorThings.csv"  
LM_IT_Label = pd.read_csv(LM_IT_csv_label, index_col = 0)  
## Merge two Dataframe to one  
LM_IT_Df = LM_Features.join(LM_IT_Label, how='inner')  
## Save to data files  
LM_IT_csv_folder_path = "/Users/user7/Downloads/HD6_12er/DataPreparation"  
LM_IT_data_name = "LM_IT_Features_Labels.csv"  
DataFrame2CSV(LM_IT_Df, LM_IT_csv_folder_path, LM_IT_data_name) 


##LC_IndoorThings
LC_IT_csv_label = "/Users/user7/Downloads/HD6_12er/DataPreparation/LC_Label_IndoorThings.csv"  
LC_IT_Label = pd.read_csv(LC_IT_csv_label, index_col = 0)  
## Merge two Dataframe to one  
LC_IT_Df = LC_Features.join(LC_IT_Label, how='inner')  
## Save to data files  
LC_IT_csv_folder_path = "/Users/user7/Downloads/HD6_12er/DataPreparation"  
LC_IT_data_name = "LC_IT_Features_Labels.csv"  
DataFrame2CSV(LC_IT_Df, LC_IT_csv_folder_path, LC_IT_data_name) 



## Append ED_IP, JY_IP, LM_IP three dataFrames into one full dataset

ED_IT = ED_IT_Df
ED_IT.shape
ED_JY_IT = ED_IT.append(JY_IT_Df)
ED_JY_IT.shape
ED_JY_LM_IT = ED_JY_IT.append(LM_IT_Df)
ED_JY_LM_IT.shape ## 192
ED_JY_LM_LC_IT = ED_JY_LM_IT.append(LC_IT_Df)
ED_JY_LM_LC_IT.shape ## 236



## save the full ED_JY_LM_df to csv
ED_JY_LM_LC_IT_path = "/Users/user7/Downloads/HD6_12er/DataPreparation"
ED_JY_LM_LC_IT_name = "ED_JY_LM_LC_IT_Features_Labels.csv"
DataFrame2CSV(ED_JY_LM_LC_IT, ED_JY_LM_LC_IT_path, ED_JY_LM_LC_IT_name)




## Work with OutdoorThings
#ED_OutdoorThings 
ED_OT_csv_label = "/Users/user7/Downloads/HD6_12er/DataPreparation/ED_Label_OutdoorThings.csv"  
ED_OT_Label = pd.read_csv(ED_OT_csv_label, index_col = 0)  
## Merge two Dataframe to one  
ED_OT_Df = ED_Features.join(ED_OT_Label, how='inner')  
## Save to data files  
ED_OT_csv_folder_path = "/Users/user7/Downloads/HD6_12er/DataPreparation"  
ED_OT_data_name = "ED_OT_Features_Labels.csv"  
DataFrame2CSV(ED_OT_Df, ED_OT_csv_folder_path, ED_OT_data_name)  
 
 
 
#JY_Outdoor Things 
JY_OT_csv_label = "/Users/user7/Downloads/HD6_12er/DataPreparation/JY_Label_OutdoorThings.csv"  
JY_OT_Label = pd.read_csv(JY_OT_csv_label, index_col = 0)  
## Merge two Dataframe to one  
JY_OT_Df = JY_Features.join(JY_OT_Label, how='inner')  
## Save to data files  
JY_OT_csv_folder_path = "/Users/user7/Downloads/HD6_12er/DataPreparation"  
JY_OT_data_name = "JY_OT_Features_Labels.csv"  
DataFrame2CSV(JY_OT_Df, JY_OT_csv_folder_path, JY_OT_data_name)  
 
 
 
# LM_Outdoor Things 
LM_OT_csv_label = "/Users/user7/Downloads/HD6_12er/DataPreparation/LM_Label_OutdoorThings.csv"  
LM_OT_Label = pd.read_csv(LM_OT_csv_label, index_col = 0)  
## Merge two Dataframe to one  
LM_OT_Df = LM_Features.join(LM_OT_Label, how='inner')  
## Save to data files  
LM_OT_csv_folder_path = "/Users/user7/Downloads/HD6_12er/DataPreparation"  
LM_OT_data_name = "LM_OT_Features_Labels.csv"  
DataFrame2CSV(LM_OT_Df, LM_OT_csv_folder_path, LM_OT_data_name)  
 
 
## LC_OutdoorThings 
LC_OT_csv_label = "/Users/user7/Downloads/HD6_12er/DataPreparation/LC_Label_OutdoorThings.csv"  
LC_OT_Label = pd.read_csv(LC_OT_csv_label, index_col = 0)  
## Merge two Dataframe to one  
LC_OT_Df = LC_Features.join(LC_OT_Label, how='inner')  
## Save to data files  
LC_OT_csv_folder_path = "/Users/user7/Downloads/HD6_12er/DataPreparation"  
LC_OT_data_name = "LC_OT_Features_Labels.csv"  
DataFrame2CSV(LC_OT_Df, LC_OT_csv_folder_path, LC_OT_data_name)  
 
 
## Append ED_IP, JY_IP, LM_IP three dataFrames into one full dataset 
 
 
ED_OT = ED_OT_Df 
ED_OT.shape 
ED_JY_OT = ED_OT.append(JY_OT_Df) 
ED_JY_OT.shape 
ED_JY_LM_OT = ED_JY_OT.append(LM_OT_Df) 
ED_JY_LM_OT.shape ## 48 
ED_JY_LM_LC_OT = ED_JY_LM_OT.append(LC_OT_Df) 
ED_JY_LM_LC_OT.shape ## 56
 
 
 
 
## save the full ED_JY_LM_df to csv 
ED_JY_LM_LC_OT_path = "/Users/user7/Downloads/HD6_12er/DataPreparation" 
ED_JY_LM_LC_OT_name = "ED_JY_LM_LC_OT_Features_Labels.csv" 
DataFrame2CSV(ED_JY_LM_LC_OT, ED_JY_LM_LC_OT_path, ED_JY_LM_LC_OT_name) 
 
 
 
 
 
 










def hist_plot(vals, lab):
    ## Distribution plot of values
    sns.distplot(vals)
    plt.title('Histogram of ' + lab)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.show()
    
#labels = np.array(auto_prices['price'])
hist_plot(auto_prices['price'], 'prices')

for col in ED_IP_Df.columns:
    hist_plot(ED_IP_Df[col], col)

## Explore the relationship between features and labels
def plot_scatter_t(auto_prices, cols, col_y, alpha = 1.0):
    for col in cols:
        fig = plt.figure(figsize=(7,6)) # define plot area
        ax = fig.gca() # define axis   
        auto_prices.plot.scatter(x = col, y = col_y, ax = ax, alpha = alpha)
        ax.set_title('Scatter plot of ' + col_y + ' vs. ' + col) # Give the plot a main title
        ax.set_xlabel(col) # Set text for the x axis
        ax.set_ylabel(col_y)# Set text for y axis
        plt.show()

Features_cols = ED_IP_Df.columns[:48]
plot_scatter_t(ED_IP_Df, Features_cols, col_y = 'Exposure', alpha= 0.8)


# Use contour plot
def plot_desity_2d(auto_prices, cols, col_y = 'price', kind ='kde'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.jointplot(col, col_y, data=auto_prices, kind=kind)
        plt.xlabel(col) # Set text for the x axis
        plt.ylabel(col_y)# Set text for y axis
        plt.show()

plot_desity_2d(ED_IP_Df, Features_cols, col_y = 'Exposure')  


# Pair wise plot for the first R5 pl
pair_cols = Features_cols.tolist()[:5]
pair_cols.append('Exposure')
pair_cols
sns.pairplot(ED_IP_Df[pair_cols], hue=None, palette="Set2", diag_kind="kde", size=2).map_upper(sns.kdeplot, cmap="Blues_d")




pair_col_R5 = Features_cols.tolist()[:5]
pair_col_G5 = Features_cols.tolist()[16:21]
pair_col_B5 = Features_cols.tolist()[32:49]
pair_col_RGB5 = []
pair_col_RGB5.extend(pair_col_R5)
pair_col_RGB5.extend(pair_col_G5)
pair_col_RGB5.extend(pair_col_B5)
pair_col_RGB5.append('Exposure')
pair_col_RGB5

sns.pairplot(ED_IP_Df[pair_col_RGB5], hue=None, palette="Set2", diag_kind="kde", size=2).map_upper(sns.kdeplot, cmap="Blues_d")


