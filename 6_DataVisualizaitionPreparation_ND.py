import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
## Load the data
IP_csv_path = "/Users/user7/Downloads/HD6_12er/DataPreparation/ED_JY_LM_LC_IP_Features_Labels.csv"

# read the file as csv and use the first row as index
IP_dataset = pd.read_csv(IP_csv_path, index_col = 0) 
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


def plot_scatter_t(auto_prices, cols, col_y, alpha = 0.1):
    for col in cols:
        fig = plt.figure(figsize=(7,6)) # define plot area
        ax = fig.gca() # define axis   
        auto_prices.plot.scatter(x = col, y = col_y, ax = ax, alpha = alpha)
        ax.set_title('Scatter plot of ' + col_y + ' vs. ' + col) # Give the plot a main title
        ax.set_xlabel(col) # Set text for the x axis
        ax.set_ylabel(col_y)# Set text for y axis
        plt.show()

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


def hist_plot(vals, lab):
    ## Distribution plot of values
    sns.distplot(vals)
    plt.title('Histogram of ' + lab)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.show()
## After review the histogram of each each column, some featues are left skewed.
hist_plot(IP_dataset_combined["R0-31"], "R0-31")

test_df = pd.DataFrame()
test_df["expR0-31"] = np.exp(IP_dataset_combined["R0-31"])
test_df.head()

hist_plot(test_df["expR0-31"], "expR0-31")
hist_plot(IP_dataset_combined["R0-31"], "R0-31")

test_df["sqrtR0-31"] = np.sqrt(IP_dataset_combined["R0-31"])
test_df.head()
hist_plot(test_df["sqrtR0-31"], "sqrtR0-31")

IP_dataset_combined_normal = pd.DataFrame()
IP_dataset_combined_normal['sqrtR0-31'] = test_df["sqrtR0-31"]

hist_plot(IP_dataset_combined["R160-191"], "R160-191")
test_df["sqrtR160-191"] = np.sqrt(IP_dataset_combined["R160-191"])
hist_plot(test_df["sqrtR160-191"], "sqrtR160-191")
IP_dataset_combined_normal["sqrtR160-191"] = test_df["sqrtR160-191"] 


#good
hist_plot(IP_dataset_combined["R192-223"], "R192-223") 
test_df["sqrtR192-223"] = np.sqrt(IP_dataset_combined["R192-223"]) 
hist_plot(test_df["sqrtR192-223"], "sqrtR192-223") 
IP_dataset_combined_normal["sqrtR192-223"] = test_df["sqrtR192-223"] 

#bad
hist_plot(IP_dataset_combined["R224-255"], "R224-255") 
test_df["sqrtR224-255"] = np.sqrt(IP_dataset_combined["R224-255"]) 
hist_plot(test_df["sqrtR224-255"], "sqrtR224-255") 
IP_dataset_combined_normal["sqrtR224-255"] = test_df["sqrtR224-255"] 
#bad
hist_plot(IP_dataset_combined["R224-255"], "R224-255")  
test_df["cbrtR224-255"] = np.cbrt(IP_dataset_combined["R224-255"])  
hist_plot(test_df["cbrtR224-255"], "cbrtR224-255")  
IP_dataset_combined_normal["cbrtR224-255"] = test_df["cbrtR224-255"] 
#good
hist_plot(IP_dataset_combined["R224-255"], "R224-255")  
test_df["powerR224-255"] = np.power(IP_dataset_combined["R224-255"], 1.0/4)  
hist_plot(test_df["powerR224-255"], "powerR224-255")  
IP_dataset_combined_normal["powerR224-255"] = test_df["powerR224-255"] 
IP_dataset_combined_normal

## power0.25 is good enough


hist_plot(IP_dataset_combined["G96-127"], "G96-127")  
test_df["sqrtG96-127"] = np.sqrt(IP_dataset_combined["G96-127"])  
hist_plot(test_df["sqrtG96-127"], "sqrtG96-127")  
IP_dataset_combined_normal["sqrtG96-127"] = test_df["sqrtG96-127"] 


hist_plot(IP_dataset_combined["G128-159"], "G128-159")  
test_df["sqrtG128-159"] = np.sqrt(IP_dataset_combined["G128-159"])  
hist_plot(test_df["sqrtG128-159"], "sqrtG128-159")  
IP_dataset_combined_normal["sqrtG128-159"] = test_df["sqrtG128-159"] 

hist_plot(IP_dataset_combined["G160-191"], "G160-191")  
test_df["sqrtG160-191"] = np.sqrt(IP_dataset_combined["G160-191"])  
hist_plot(test_df["sqrtG160-191"], "sqrtG160-191")  
IP_dataset_combined_normal["sqrtG160-191"] = test_df["sqrtG160-191"] 

hist_plot(IP_dataset_combined["G192-223"], "G192-223")  
test_df["sqrtG192-223"] = np.sqrt(IP_dataset_combined["G192-223"])  
hist_plot(test_df["sqrtG192-223"], "sqrtG192-223")  
IP_dataset_combined_normal["sqrtG192-223"] = test_df["sqrtG192-223"] 


#bad
hist_plot(IP_dataset_combined["G224-255"], "G224-255")  
test_df["sqrtG224-255"] = np.sqrt(IP_dataset_combined["G224-255"])  
hist_plot(test_df["sqrtG224-255"], "sqrtG224-255")  
IP_dataset_combined_normal["sqrtG224-255"] = test_df["sqrtG224-255"] 
#good
hist_plot(IP_dataset_combined["G224-255"], "G224-255")  
test_df["powerG224-255"] = np.power(IP_dataset_combined["G224-255"], 1.0/4)  
hist_plot(test_df["powerG224-255"], "powerG224-255")  
IP_dataset_combined_normal["powerG224-255"] = test_df["powerG224-255"] 


 
hist_plot(IP_dataset_combined["B96-127"], "B96-127")  
test_df["sqrtB96-127"] = np.sqrt(IP_dataset_combined["B96-127"])  
hist_plot(test_df["sqrtB96-127"], "sqrtB96-127")  
IP_dataset_combined_normal["sqrtB96-127"] = test_df["sqrtB96-127"] 

hist_plot(IP_dataset_combined["B128-159"], "B128-159")  
test_df["sqrtB128-159"] = np.sqrt(IP_dataset_combined["B128-159"])  
hist_plot(test_df["sqrtB128-159"], "sqrtB128-159")  
IP_dataset_combined_normal["sqrtB128-159"] = test_df["sqrtB128-159"] 

hist_plot(IP_dataset_combined["B160-191"], "B160-191")  
test_df["sqrtB160-191"] = np.sqrt(IP_dataset_combined["B160-191"])  
hist_plot(test_df["sqrtB160-191"], "sqrtB160-191")  
IP_dataset_combined_normal["sqrtB160-191"] = test_df["sqrtB160-191"] 

#bad
hist_plot(IP_dataset_combined["B192-223"], "B192-223")  
test_df["sqrtB192-223"] = np.sqrt(IP_dataset_combined["B192-223"])  
hist_plot(test_df["sqrtB192-223"], "sqrtB192-223")  
IP_dataset_combined_normal["sqrtB192-223"] = test_df["sqrtB192-223"] 
#good
hist_plot(IP_dataset_combined["B192-223"], "B192-223")  
test_df["powerB192-223"] = np.power(IP_dataset_combined["B192-223"], 1.0/4)  
hist_plot(test_df["powerB192-223"], "powerB192-223")  
IP_dataset_combined_normal["powerB192-223"] = test_df["powerB192-223"] 

#bad
hist_plot(IP_dataset_combined["B224-255"], "B224-255")  
test_df["sqrtB224-255"] = np.sqrt(IP_dataset_combined["B224-255"])  
hist_plot(test_df["sqrtB224-255"], "sqrtB224-255")  
IP_dataset_combined_normal["sqrtB224-255"] = test_df["sqrtB224-255"] 
#good
hist_plot(IP_dataset_combined["B224-255"], "B224-255")  
test_df["powerB224-255"] = np.power(IP_dataset_combined["B224-255"], 1.0/4)  
hist_plot(test_df["powerB224-255"], "powerB224-255")  
IP_dataset_combined_normal["powerB224-255"] = test_df["powerB224-255"] 


IP_dataset_combined_normal.head()
IP_dataset_combined_normal_col = IP_dataset_combined_normal.columns.tolist()
IP_dataset_combined_normal_col

IP_dataset_combined.head()
IP_dataset_combined_col = IP_dataset_combined.columns.tolist()
IP_dataset_combined_col

IT_dataset_combined_ND = IP_dataset_combined.join(IP_dataset_combined_normal, how="inner")
IT_dataset_combined_ND.head()
cols = IT_dataset_combined_ND.columns.tolist()
cols

new_cols = ['sqrtR0-31','R32-63','R64-95',
 'R96-127',
 'R128-159',
 'sqrtR160-191',
 'sqrtR192-223',
 'powerR224-255',
 'G0-31',
 'G32-63',
 'G64-95',
 'sqrtG96-127',
 'sqrtG128-159',
 'sqrtG160-191',
 'sqrtG192-223',
 'powerG224-255',
 'B0-31',
 'B32-63',
 'B64-95',
 'sqrtB96-127',
 'sqrtB128-159',
 'sqrtB160-191',
 'powerB192-223',
 'powerB224-255',
 'Exposure',
 'Contrast',
 'Highlights',
 'Shadows',
 'Temperature']
new_IT_dataset_combine_ND = IT_dataset_combined_ND[new_cols]
new_IT_dataset_combine_ND.head()

for col in new_IT_dataset_combine_ND.columns:
    hist_plot(new_IT_dataset_combine_ND[col], col)


sns.pairplot(new_IT_dataset_combine_ND[new_cols], hue=None, palette="Set2", diag_kind="kde", size=2).map_upper(sns.kdeplot, cmap="Blues_d")


def DataFrame2CSV(df, csv_folder_path, csv_file_name):
    
    csv_file_path = os.path.join(csv_folder_path, csv_file_name)
    df.to_csv(csv_file_path)
new_IT_dataset_combine_ND_path= "/Users/user7/Downloads/HD6_12er/DataPreparation"
new_IT_dataset_combine_ND_name = "new_IT_dataset_combine_ND.csv"
DataFrame2CSV(new_IT_dataset_combine_ND, new_IT_dataset_combine_ND_path, new_IT_dataset_combine_ND_name)