import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
## Load the data
IP_csv_path = "/Users/user7/Downloads/HD6_12er/DataPreparation/ED_JY_LM_LC_IP_Features_Labels.csv"

# read the file as csv and use the first row as index
IP_dataset = pd.read_csv(IP_csv_path, index_col = 0) 
IP_dataset # 1609 X 53

# Explore the data
IP_dataset.columns
IP_dataset.describe()

# Plot hist_plot of numerical colomns
def hist_plot(vals, lab):
    ## Distribution plot of values
    sns.distplot(vals)
    plt.title('Histogram of ' + lab)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.show()
    
hist_plot(IP_dataset['Exposure'], 'Exposure')

for col in IP_dataset.columns:
    hist_plot(IP_dataset[col], col)

IP_dataset.head()

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

# Features vs. Exposure
Features_cols = IP_dataset.columns[:48]
plot_scatter_t(IP_dataset, Features_cols, col_y = 'Exposure', alpha= 0.1)

# Features vs. Contrast
Features_cols = IP_dataset.columns[:48]
plot_scatter_t(IP_dataset, Features_cols, col_y = 'Contrast', alpha= 0.1)


# Features vs. Highlights
Features_cols = IP_dataset.columns[:48]
plot_scatter_t(IP_dataset, Features_cols, col_y = 'Highlights', alpha= 0.1)

# Features vs. Shadows
Features_cols = IP_dataset.columns[:48]
plot_scatter_t(IP_dataset, Features_cols, col_y = 'Shadows', alpha= 0.1)


# Use contour plot
def plot_desity_2d(auto_prices, cols, col_y = 'price', kind ='kde'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.jointplot(col, col_y, data=auto_prices, kind=kind)
        plt.xlabel(col) # Set text for the x axis
        plt.ylabel(col_y)# Set text for y axis
        plt.show()

#plot_desity_2d(IP_dataset, Features_cols, col_y = 'Exposure')  


# Pair wise plot for the R channel
pair_cols = Features_cols.tolist()[:16]
#pair_cols.append('Exposure')
pair_cols
sns.pairplot(IP_dataset[pair_cols], hue=None, palette="Set2", diag_kind="kde", size=2).map_upper(sns.kdeplot, cmap="Blues_d")

# Pair wise plot for the G channel
pair_cols = Features_cols.tolist()[16:32]
#pair_cols.append('Exposure')
pair_cols
sns.pairplot(IP_dataset[pair_cols], hue=None, palette="Set2", diag_kind="kde", size=2).map_upper(sns.kdeplot, cmap="Blues_d")

# Pair wise plot for the B channel
pair_cols = Features_cols.tolist()[32:48]
#pair_cols.append('Exposure')
pair_cols
sns.pairplot(IP_dataset[pair_cols], hue=None, palette="Set2", diag_kind="kde", size=2).map_upper(sns.kdeplot, cmap="Blues_d")


### Since the new bins seems to belinear, combine them
IP_columns = IP_dataset.columns.tolist()
IP_columns

IP_dataset_combined = pd.DataFrame()

# Set the Features Column names
Features_Columns = []
Columns_color = ['R', 'G', 'B']
Columns_list = []

for color in Columns_color:
    k = 0
    for i in range(8):
        col_name = color + str(k) + "-" + str(k+31)
        Features_Columns.append(col_name)
        k = k + 32

Features_Columns    
k=0
for i in range(24):
    IP_dataset_combined[Features_Columns[i]] = IP_dataset[IP_columns[k]] +IP_dataset[IP_columns[k+1]]
    k = k + 2
IP_dataset_combined[['Exposure','Contrast','Highlights','Shadows','Temperature']] = IP_dataset[['Exposure','Contrast','Highlights','Shadows','Temperature']]
IP_dataset_combined.head()
IP_dataset.head()


## Now the dataframe is updated, plot to review again.
for col in IP_dataset_combined.columns:
    hist_plot(IP_dataset_combined[col], col)

## Run the pair plot
sns.pairplot(IP_dataset_combined[Features_Columns], hue=None, palette="Set2", diag_kind="kde", size=2).map_upper(sns.kdeplot, cmap="Blues_d")



## View the Features vs. Labels scatter plot seperately
# Features vs. Exposure
Features_cols = IP_dataset_combined.columns[:24]
plot_scatter_t(IP_dataset_combined, Features_cols, col_y = 'Exposure', alpha= 0.3)

# Features vs. Contrast
Features_cols = IP_dataset_combined.columns[:24]
plot_scatter_t(IP_dataset_combined, Features_cols, col_y = 'Contrast', alpha= 0.3)


# Features vs. Highlights
Features_cols = IP_dataset_combined.columns[:24]
plot_scatter_t(IP_dataset_combined, Features_cols, col_y = 'Highlights', alpha= 0.3)

# Features vs. Shadows
Features_cols = IP_dataset_combined.columns[:24]
plot_scatter_t(IP_dataset_combined, Features_cols, col_y = 'Shadows', alpha= 0.3)


## After review the histogram of each each column, some featues are left skewed.
hist_plot(IP_dataset_combined[col], col)



pair_cols = Features_cols.tolist()[:5]
pair_cols.append('Contrast')
pair_cols
sns.pairplot(IP_dataset[pair_cols], hue=None, palette="Set2", diag_kind="kde", size=2).map_upper(sns.kdeplot, cmap="Blues_d")


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


