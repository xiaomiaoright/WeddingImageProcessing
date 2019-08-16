import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

## Load the ED_JY_LM_LC_IP_Features_Labels dataset
IP_csv_path = "/Users/user7/Downloads/HD6_12er/DataPreparation/ED_JY_LM_LC_IP_Features_Labels.csv"

# read the file as csv and use the first row as index
IP_dataset = pd.read_csv(IP_csv_path, index_col = 0) 
IP_columns = IP_dataset.columns.tolist()
IP_columns

# plot the histogram of all features and labels
# hist_plot_dataset
def hist_plot_dataset(dataset, col):
    ## Distribution plot of values
    sns.distplot(dataset[col])
    plt.title('Histogram of ' + col)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.show()

## Set the function need to describe the result
def print_metrics(y_true, y_predicted):
    ## First compute R^2 and the adjusted R^2
    r2 = sklm.r2_score(y_true, y_predicted)
    
    ## Print the usual metrics and the R^2 values
    print('Mean Square Error      = ' + str(sklm.mean_squared_error(y_true, y_predicted)))
    print('Root Mean Square Error = ' + str(math.sqrt(sklm.mean_squared_error(y_true, y_predicted))))
    print('Mean Absolute Error    = ' + str(sklm.mean_absolute_error(y_true, y_predicted)))
    print('Median Absolute Error  = ' + str(sklm.median_absolute_error(y_true, y_predicted)))
    print('R^2                    = ' + str(r2))
    
def resid_plot(y_test, y_score):
    ## first compute vector of residuals. 
    resids = np.subtract(y_test.reshape(-1,1), y_score.reshape(-1,1))
    ## now make the residual plots
    sns.regplot(y_score, resids, fit_reg=False)
    plt.title('Residuals vs. predicted values')
    plt.xlabel('Predicted values')
    plt.ylabel('Residual')
    plt.show()

def hist_resids(y_test, y_score):
    ## first compute vector of residuals. 
    resids = np.subtract(y_test.reshape(-1,1), y_score.reshape(-1,1))
    ## now make the residual plots
    sns.distplot(resids)
    plt.title('Histogram of residuals')
    plt.xlabel('Residual value')
    plt.ylabel('count')
    plt.show()
    
def resid_qq(y_test, y_score):
    ## first compute vector of residuals. 
    resids = np.subtract(y_test, y_score)
    ## now make the residual plots
    ss.probplot(resids.flatten(), plot = plt)
    plt.title('Residuals vs. predicted values')
    plt.xlabel('Predicted values')
    plt.ylabel('Residual')
    plt.show()

for col in IP_columns:
        hist_plot_dataset(IP_dataset, col)
## The result shows that some of the  features are right skewed. So feature engineering to make it linear
# save the transformed result to a temp dataframe
# define a new dataframe

temp_IP_dataset = pd.DataFrame()

# find the right skewed column names
right_skewed_cols = ["R0-15", 'R176-191', 'R192-207', 'R208-223','R224-239', 'R240-255','G0-15',
 'G96-111','G112-127','G128-143','G144-159','G160-175','G176-191','G192-207','G208-223',
 'G224-239','G240-255','B48-63','B64-79','B80-95','B96-111','B112-127','B128-143','B144-159',
'B160-175','B176-191','B192-207','B208-223','B224-239','B240-255']
normal_distributed_cols = ['R16-31','R32-47','R48-63','R64-79','R80-95','R96-111','R112-127','R128-143','R144-159','R160-175',
'G16-31','G32-47','G48-63','G64-79','G80-95','B0-15','B16-31','B32-47']
label_cols = ['Exposure','Contrast','Highlights', 'Shadows', 'Temperature']


# define the fixed skew col names
fixed_skew_cols = []
for col in right_skewed_cols:
        col_new = "sqr" + col
        fixed_skew_cols.append(col_new)

# save the transformed columns to temp dataset
temp_IP_dataset[fixed_skew_cols] = np.sqrt(IP_dataset[right_skewed_cols])
temp_IP_dataset.shape

# view the distribution after the transformation
for col in temp_IP_dataset.columns:
        hist_plot_dataset(temp_IP_dataset, col)

# The result shows some of the columns are still right skewed, but gets better now.
# Try to use the updated dataset to test the models
IP_dataset_FixedSkew = temp_IP_dataset.join(IP_dataset[normal_distributed_cols], how = 'inner')
IP_dataset_FixedSkew = IP_dataset_FixedSkew.join(IP_dataset[label_cols])
IP_dataset_FixedSkew.head()

## Uset he IP_dataset_FixedSkew dataset to test the NN models
# set Feature array and labels array
Features = np.array(IP_dataset_FixedSkew)[:,:48]
Labels_exposure = np.array(IP_dataset_FixedSkew)[:,48]
Labels_contrast = np.array(IP_dataset_FixedSkew)[:,49]
Labels_highlights = np.array(IP_dataset_FixedSkew)[:,50]
Labels_shadows = np.array(IP_dataset_FixedSkew)[:,51]

# print the shape of the features and label array
print(Features.shape)
print(Labels_exposure.shape)

# Split train and test set
X_train, X_test, y_train, y_test = train_test_split(Features, Labels_exposure, test_size=0.3, random_state=1122)

## Normalize the features
scaler = preprocessing.StandardScaler().fit(X_train) # use MinMaxScaler() because all datapoints are between 0-1
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


### Step2: Build the NN model
# define the model
NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(128, kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()

"""
# Set checkpoints
checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]
"""

# Train the model 
NN_model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)

# Predict the test set to evaluate the model
y_score = NN_model.predict(X_test)

# Print the evluation Results and metrics
print_metrics(y_test, y_score)
hist_resids(y_test, y_score)  
resid_qq(y_test, y_score) 
resid_plot(y_test, y_score) 
# The result is slightly better than using the original dataset
"""
Fitting result with IP_dataset_FixedSkew
Mean Square Error      = 0.1426174385854945
Root Mean Square Error = 0.37764724093457175
Mean Absolute Error    = 0.26180579928507713
Median Absolute Error  = 0.17931842803955078
R^2                    = 0.4210641980381006
"""


"""
The fitting result using the orginal IP_dataset
Mean Square Error      = 0.15969673549976218
Root Mean Square Error = 0.3996207395766168
Mean Absolute Error    = 0.28032845768992704
Median Absolute Error  = 0.18909698724746704
R^2                    = 0.3517331502077923
"""


### Since there still are some of the columns are right skewed in IP_DataSet_FixedSkew. Try to fixed those columsn
# veiew the histgram of IP_DataSet_FixedSkew 
for col in IP_dataset_FixedSkew.columns:
        hist_plot_dataset(IP_dataset_FixedSkew, col)

IP_dataset_FixedSkew.columns

right_skewed_cols_2 = ['sqrR240-255','sqrG176-191','sqrG192-207','sqrG208-223', 'sqrG224-239', 'sqrG240-255','sqrB192-207','sqrB208-223','sqrB224-239', 'sqrB240-255']
normal_distributed_cols_2 = ['sqrR0-15', 'sqrR176-191', 'sqrR192-207', 'sqrR208-223', 'sqrR224-239',
'sqrG0-15', 'sqrG96-111', 'sqrG112-127', 'sqrG128-143',
'sqrG144-159', 'sqrG160-175', 'sqrB48-63', 'sqrB64-79',
'sqrB80-95', 'sqrB96-111', 'sqrB112-127', 'sqrB128-143', 'sqrB144-159',
'sqrB160-175', 'sqrB176-191',   'R16-31', 'R32-47', 'R48-63', 'R64-79',
'R80-95', 'R96-111', 'R112-127', 'R128-143', 'R144-159', 'R160-175',
'G16-31', 'G32-47', 'G48-63', 'G64-79', 'G80-95', 'B0-15', 'B16-31','B32-47']

       
       
        
       
       
       
       
       


# R0-15 channel
hist_plot(IP_dataset["R192-223"], "R192-223") 
test_df["sqrtR192-223"] = np.sqrt(IP_dataset["R192-223"]) 
hist_plot(test_df["sqrtR192-223"], "sqrtR192-223") 
IP_dataset_combined_normal["sqrtR192-223"] = test_df["sqrtR192-223"] 

















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