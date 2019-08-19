import math
## Try 32 bins
## save the IP_dataset_FixedSkew to csv
import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as nr
# Check the distribution of features
import pandas as pd
import rawpy
import scipy.stats as ss
import seaborn as sns
## sklearn GPR regressor cannot perform feature selections
#define the model
import sklearn
import sklearn.metrics as sklm
import sklearn.model_selection as ms
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Dense, Flatten
from keras.models import Sequential
from matplotlib import pyplot as plt
from pytictoc import TicToc
## Define the linear regression model
from sklearn import feature_selection as fs
from sklearn import gaussian_process, linear_model
from sklearn import metrics as sklm
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.neural_network import MLPRegressor


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
Fitting result with IP_dataset_FixedSkew, NN Models
Mean Square Error      = 0.1426174385854945
Root Mean Square Error = 0.37764724093457175
Mean Absolute Error    = 0.26180579928507713
Median Absolute Error  = 0.17931842803955078
R^2                    = 0.4210641980381006




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

right_skewed_cols2 = ['sqrR240-255','sqrG176-191','sqrG192-207','sqrG208-223', 'sqrG224-239', 'sqrG240-255','sqrB192-207','sqrB208-223','sqrB224-239', 'sqrB240-255']
normal_distributed_cols2 = ['sqrR0-15', 'sqrR176-191', 'sqrR192-207', 'sqrR208-223', 'sqrR224-239',
'sqrG0-15', 'sqrG96-111', 'sqrG112-127', 'sqrG128-143',
'sqrG144-159', 'sqrG160-175', 'sqrB48-63', 'sqrB64-79',
'sqrB80-95', 'sqrB96-111', 'sqrB112-127', 'sqrB128-143', 'sqrB144-159',
'sqrB160-175', 'sqrB176-191',   'R16-31', 'R32-47', 'R48-63', 'R64-79',
'R80-95', 'R96-111', 'R112-127', 'R128-143', 'R144-159', 'R160-175',
'G16-31', 'G32-47', 'G48-63', 'G64-79', 'G80-95', 'B0-15', 'B16-31','B32-47']
label_cols = ['Exposure','Contrast','Highlights', 'Shadows', 'Temperature']

temp_IP_dataset2 = pd.DataFrame()

# define the fixed skew col names
fixed_skew_cols2 = []
for col in right_skewed_cols2:
        col_new = "sqr2" + col
        fixed_skew_cols2.append(col_new)
fixed_skew_cols2

# save the transformed columns to temp dataset
temp_IP_dataset2[fixed_skew_cols2] = np.sqrt(IP_dataset_FixedSkew[right_skewed_cols2])
temp_IP_dataset2.head()     

# view the distribution after the transformation
for col in temp_IP_dataset2.columns:
        hist_plot_dataset(temp_IP_dataset2, col)


 # The result shows some of the columns are still right skewed, but gets better now.
# Try to use the updated dataset to test the models
IP_dataset_FixedSkew2 = temp_IP_dataset2.join(IP_dataset_FixedSkew[normal_distributed_cols2], how = 'inner')
IP_dataset_FixedSkew2 = IP_dataset_FixedSkew2.join(IP_dataset_FixedSkew[label_cols])
IP_dataset_FixedSkew2.head()
IP_dataset_FixedSkew2.shape

## Test the result with NN models

# set Feature array and labels array
Features = np.array(IP_dataset_FixedSkew2)[:,:48]
Labels_exposure = np.array(IP_dataset_FixedSkew2)[:,48]
Labels_contrast = np.array(IP_dataset_FixedSkew2)[:,49]
Labels_highlights = np.array(IP_dataset_FixedSkew2)[:,50]
Labels_shadows = np.array(IP_dataset_FixedSkew2)[:,51]

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

## Compare the result with original dataset, IP_dataset_FixedSkew dataset
## The result shows that with IP_Dataset_FixedSkew2, the NN seems to work worse.
"""
Fitting result with IP_dataset_FixedSkew2
Mean Square Error      = 0.15232582886750226
Root Mean Square Error = 0.3902894168018168
Mean Absolute Error    = 0.27399471800273006
Median Absolute Error  = 0.19796121716499326
R^2                    = 0.3816543280431073


Fitting result with IP_dataset_FixedSkew
Mean Square Error      = 0.1426174385854945
Root Mean Square Error = 0.37764724093457175
Mean Absolute Error    = 0.26180579928507713
Median Absolute Error  = 0.17931842803955078
R^2                    = 0.4210641980381006




The fitting result using the orginal IP_dataset
Mean Square Error      = 0.15969673549976218
Root Mean Square Error = 0.3996207395766168
Mean Absolute Error    = 0.28032845768992704
Median Absolute Error  = 0.18909698724746704
R^2                    = 0.3517331502077923
"""

## So I will use the IP_dataset_FixedSkew dataset, and try feature selection
# Try with linear regression models with IP_dataset_FixedSkew first
## Uset he IP_dataset_FixedSkew dataset to test the NN models
# set Feature array and labels array
Features = np.array(IP_dataset_FixedSkew)[:,:48]
Labels_exposure = np.array(IP_dataset_FixedSkew)[:,48]
Labels_contrast = np.array(IP_dataset_FixedSkew)[:,49]
Labels_highlights = np.array(IP_dataset_FixedSkew)[:,50]
Labels_shadows = np.array(IP_dataset_FixedSkew)[:,51]

# Split train and test set
X_train, X_test, y_train, y_test = train_test_split(Features, Labels_exposure, test_size=0.3, random_state=1122)

## Normalize the features
scaler = preprocessing.StandardScaler().fit(X_train) # use MinMaxScaler() because all datapoints are between 0-1
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

linear_mod = linear_model.LinearRegression()
linear_mod.fit(X_train, y_train)

# predict the y_score
y_score = linear_mod.predict(X_test)

# Print the evluation Results and metrics
print_metrics(y_test, y_score)
hist_resids(y_test, y_score)  
resid_qq(y_test, y_score) 
resid_plot(y_test, y_score) 

"""
worse than neural networks
Fitting result for linear regressiong model with IP_dataset_FixedSkew dataset
Mean Square Error      = 0.1897046076394065
Root Mean Square Error = 0.4355509242779844
Mean Absolute Error    = 0.3383098395351773
Median Absolute Error  = 0.27236359506020436
R^2                    = 0.2299203361884128
"""

### Try l1 regulations

## Define the linear regression model
linear_mod_l1 = linear_model.Lasso(alpha = 0.005)
linear_mod_l1.fit(X_train, y_train)

# predict the y_score
y_score = linear_mod_l1.predict(X_test)

# Print the evluation Results and metrics
print_metrics(y_test, y_score)
hist_resids(y_test, y_score)  
resid_qq(y_test, y_score) 
resid_plot(y_test, y_score)

"""
The result is slightly better than the linear regression, but still worse than NN
Mean Square Error      = 0.18794413512184202
Root Mean Square Error = 0.4335252416201876
Mean Absolute Error    = 0.33847422602762284
Median Absolute Error  = 0.2689085398868647
R^2                    = 0.23706673132000877
"""
       
### Try with alpha = 0.0005
## Define the linear regression model
linear_mod_l1 = linear_model.Lasso(alpha = 0.0005)
linear_mod_l1.fit(X_train, y_train)

# predict the y_score
y_score = linear_mod_l1.predict(X_test)

# Print the evluation Results and metrics
print_metrics(y_test, y_score)
hist_resids(y_test, y_score)  
resid_qq(y_test, y_score) 
resid_plot(y_test, y_score)
       
"""
### Try with alpha = 0.0005... no difference
Mean Square Error      = 0.1872593059071346
Root Mean Square Error = 0.43273468304162377
Mean Absolute Error    = 0.33614430478876206
Median Absolute Error  = 0.27169005125623846
R^2                    = 0.23984670096857252
"""


### Try a different NN, with 6 hidden layers, 100 epochs
### Step2: Build the NN model
# define the model
NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(128, kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
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
NN_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)

# Predict the test set to evaluate the model
y_score = NN_model.predict(X_test)

# Print the evluation Results and metrics
print_metrics(y_test, y_score)
hist_resids(y_test, y_score)  
resid_qq(y_test, y_score) 
resid_plot(y_test, y_score) 

"""
Fitting result with IP_dataset_FixedSkew NN, 6 hidden layer 100 epcho
Mean Square Error      = 0.1534899725918332
Root Mean Square Error = 0.3917779633821091
Mean Absolute Error    = 0.27238959785897904
Median Absolute Error  = 0.18986990451812746
R^2                    = 0.3769286473176017


Fitting result with IP_dataset_FixedSkew NN, 3 hidden layer 500 epcho
Mean Square Error      = 0.1426174385854945
Root Mean Square Error = 0.37764724093457175
Mean Absolute Error    = 0.26180579928507713
Median Absolute Error  = 0.17931842803955078
R^2                    = 0.4210641980381006




The fitting result using the orginal IP_dataset
Mean Square Error      = 0.15969673549976218
Root Mean Square Error = 0.3996207395766168
Mean Absolute Error    = 0.28032845768992704
Median Absolute Error  = 0.18909698724746704
R^2                    = 0.3517331502077923

"""


### Now try model selection
## Define the variance threhold and fit the threshold to the feature array. 
sel = fs.VarianceThreshold(threshold=0.16)
Features_reduced = sel.fit_transform(X_train)

## Print the support and shape for the transformed features
print(sel.get_support())
print(Features_reduced.shape)

# select k best feature
## Reshape the Label array, cannot perform selection on NN

### Try to use CNN to predict the result
# set Feature array and labels array
Features = np.array(IP_dataset_FixedSkew)[:,:48]
Labels_exposure = np.array(IP_dataset_FixedSkew)[:,48]
Labels_contrast = np.array(IP_dataset_FixedSkew)[:,49]
Labels_highlights = np.array(IP_dataset_FixedSkew)[:,50]
Labels_shadows = np.array(IP_dataset_FixedSkew)[:,51]

# Split train and test set
X_train, X_test, y_train, y_test = train_test_split(Features, Labels_exposure, test_size=0.3, random_state=1122)

IP_dataset_FixedSkew.columns



def DataFrame2CSV(df, csv_folder_path, csv_file_name):
    
    csv_file_path = os.path.join(csv_folder_path, csv_file_name)
    df.to_csv(csv_file_path)
csv_folder_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDataPrep"
csv_file_name = "IP_dataset_FixedSkew.csv"
DataFrame2CSV(IP_dataset_FixedSkew, csv_folder_path, csv_file_name)

## View the relationship between features and label
cols_features = IP_dataset_FixedSkew.columns.tolist()[:48]

# View the features vs. exposure
plot_scatter_t(IP_dataset_FixedSkew, cols_features, "Exposure", alpha =0.1)


def plot_scatter_t(auto_prices, cols, col_y, alpha = 0.1):
    for col in cols:
        fig = plt.figure(figsize=(7,6)) # define plot area
        ax = fig.gca() # define axis   
        auto_prices.plot.scatter(x = col, y = col_y, ax = ax, alpha = alpha)
        ax.set_title('Scatter plot of ' + col_y + ' vs. ' + col) # Give the plot a main title
        ax.set_xlabel(col) # Set text for the x axis
        ax.set_ylabel(col_y)# Set text for y axis
        plt.show()

## No clear relationship between features and exposure

### Try combine bins 16 -->> 8 with orginial dataset

original_columns = IP_dataset.columns[:48]
original_columns
['R0-15', 'R16-31', 'R32-47', 'R48-63', 'R64-79', 'R80-95', 'R96-111',
       'R112-127', 'R128-143', 'R144-159', 'R160-175', 'R176-191', 'R192-207',
       'R208-223', 'R224-239', 'R240-255', 'G0-15', 'G16-31', 'G32-47',
       'G48-63', 'G64-79', 'G80-95', 'G96-111', 'G112-127', 'G128-143',
       'G144-159', 'G160-175', 'G176-191', 'G192-207', 'G208-223', 'G224-239',
       'G240-255', 'B0-15', 'B16-31', 'B32-47', 'B48-63', 'B64-79', 'B80-95',
       'B96-111', 'B112-127', 'B128-143', 'B144-159', 'B160-175', 'B176-191',
       'B192-207', 'B208-223', 'B224-239', 'B240-255']

new_columns = []
k = 0
for i in range(24):
        new_col_name = original_columns[k].split("-")[0]+"_"+original_columns[k+1].split("-")[1]
        new_columns.append(new_col_name)
        k = k+2
new_columns


IP_dataset_8bins = pd.DataFrame()
k = 0
for i in range(len(new_columns)):
        IP_dataset_8bins[new_columns[i]] = IP_dataset[original_columns[k]] + IP_dataset[original_columns[k+1]] 
        k = k+2
IP_dataset_8bins.head()
IP_dataset.head()

## Now the IP_dataset_8bins has 24 columns as features. Join the labels columns
IP_dataset_8bins = IP_dataset_8bins.join(IP_dataset[label_cols], how = 'inner')
IP_dataset_8bins.head()

## View the distribution of IP_dataset_8Bins
for col in IP_dataset_8bins.columns:
        hist_plot_dataset(IP_dataset_8bins,col )

## Some of the columns are right skewed, but just try the NN models first
# set Feature array and labels array
Features = np.array(IP_dataset_8bins)[:,:24]
Labels_exposure = np.array(IP_dataset_8bins)[:,24]
Labels_contrast = np.array(IP_dataset_8bins)[:,25]
Labels_highlights = np.array(IP_dataset_8bins)[:,26]
Labels_shadows = np.array(IP_dataset_8bins)[:,27]

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
NN_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)

# Predict the test set to evaluate the model
y_score = NN_model.predict(X_test)

# Print the evluation Results and metrics
print_metrics(y_test, y_score)
hist_resids(y_test, y_score)  
resid_qq(y_test, y_score) 
resid_plot(y_test, y_score) 

"""
Fitting result with IP_dataset_8bins, NN, 3 hidden layer, 100 epcho
Mean Square Error      = 0.176819508300568
Root Mean Square Error = 0.4204991180734723
Mean Absolute Error    = 0.3043008799557854
Median Absolute Error  = 0.22199698686599734
R^2                    = 0.28222561801842816

Fitting result with IP_dataset_FixedSkew NN, 6 hidden layer 100 epcho
Mean Square Error      = 0.1534899725918332
Root Mean Square Error = 0.3917779633821091
Mean Absolute Error    = 0.27238959785897904
Median Absolute Error  = 0.18986990451812746
R^2                    = 0.3769286473176017


Fitting result with IP_dataset_FixedSkew NN, 3 hidden layer 500 epcho
Mean Square Error      = 0.1426174385854945
Root Mean Square Error = 0.37764724093457175
Mean Absolute Error    = 0.26180579928507713
Median Absolute Error  = 0.17931842803955078
R^2                    = 0.4210641980381006




The fitting result using the orginal IP_dataset, NN, now skew correction
Mean Square Error      = 0.15969673549976218
Root Mean Square Error = 0.3996207395766168
Mean Absolute Error    = 0.28032845768992704
Median Absolute Error  = 0.18909698724746704
R^2                    = 0.3517331502077923
"""

### Try from sklearn.neural_network import MLPRegressor

## Load the ED_JY_LM_LC_IP_Features_Labels dataset
IP_csv_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDataPrep/IP_dataset_FixedSkew.csv"

# read the file as csv and use the first row as index
IP_dataset_FixedSkew = pd.read_csv(IP_csv_path, index_col = 0) 
IP_dataset_FixedSkew.columns

## Uset he IP_dataset_FixedSkew dataset to test the sklearn NN regressor models
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

nr.seed(1115)
nn_mod = MLPRegressor(hidden_layer_sizes = (70,))
nn_mod.fit(X_train, y_train)
y_score = nn_mod.predict(X_test)

# Print the evluation Results and metrics
print_metrics(y_test, y_score)
hist_resids(y_test, y_score)  
resid_qq(y_test, y_score) 
resid_plot(y_test, y_score) 



"""
Mean Square Error      = 0.18806224048504577
Root Mean Square Error = 0.4336614353214334
Mean Absolute Error    = 0.323565202228712
Median Absolute Error  = 0.2493375818579977
R^2                    = 0.23658729890388552
"""

## Define the variance threhold and fit the threshold to the feature array. 

sel = fs.VarianceThreshold(threshold=0.16)
Features_reduced = sel.fit_transform(X_train)

# All features are good
## Print the support and shape for the transformed features

print(sel.get_support())
print(Features_reduced.shape)


# select k best feature
## Reshape the Label array
Labels = y_train
Labels = Labels.reshape(Labels.shape[0],)
## Set folds for nested cross validation

nr.seed(988)

feature_folds = ms.KFold(n_splits=10, shuffle = True)

## Define the model
nn_mod = MLPRegressor(hidden_layer_sizes = (70,))

## Perform feature selection by CV with high variance features only

nr.seed(6677)
selector = fs.RFECV(estimator = nn_mod, cv = feature_folds) # scoring = sklearn.metrics.r2_score
selector = selector.fit(Features_reduced, Labels)
selector.support_ 
selector.ranking_
Features_reduced = selector.transform(Features_reduced)

Features_reduced.shape
# Transform X_test

X_test_reduced = selector.transform(X_test)
plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
plt.title('Mean AUC by number of features')
plt.ylabel('AUC')
plt.xlabel('Number of features')

gpr_mod = sklearn.gaussian_process.GaussianProcessRegressor()

kernel = DotProduct() + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kernel,random_state=0).fit(X_train, y_train)
gpr.score(X_train, y_train) 
gpr.score(X_test, y_test)

y_score = gpr.predict(X_test, return_std=True) 
len(y_score)



# Print the evluation Results and metrics
print_metrics(y_test, y_score)
hist_resids(y_test, y_score)  
resid_qq(y_test, y_score) 
resid_plot(y_test, y_score) 




## Decided to tried build features with bins=32









def NEFFolder2Feature32bins(NEF_folder_path):
    Features = pd.DataFrame()

    # go through all XMP files in folder
    files = sorted(os.listdir(NEF_folder_path))
    

    for file_idx in range(len(files)):
        feature_list = []

        NEF_file_path = os.path.join(NEF_folder_path,files[file_idx])

        rp_image = rawpy.imread(NEF_file_path)
        rgb = rp_image.postprocess()

        # red the R, G, B channels seperately and sort from 0 to 255
        rgb_1 = np.sort(rgb[:,:,0].ravel())
        rgb_2 = np.sort(rgb[:,:,1].ravel())
        rgb_3 = np.sort(rgb[:,:,2].ravel())

        dataset = pd.DataFrame({'Red': rgb_1, 'Green': rgb_2, 'Blue':rgb_3})

        df_1 = dataset['Red'].value_counts(bins = 32, normalize = True).sort_index()
        df_2 = dataset['Green'].value_counts(bins = 32, normalize = True).sort_index()
        df_3 = dataset['Blue'].value_counts(bins = 32, normalize = True).sort_index()

        feature_list.append(df_1.tolist())
        feature_list.append(df_2.tolist())
        feature_list.append(df_3.tolist())

        feature_array = np.array(feature_list).ravel()

        Features[files[file_idx].split(".")[0]] = feature_array

    return Features.T

LM_NEF = "/Users/user7/Downloads/HD6_12er/Lauren & Matt Wedding/2017-09-08/LM_NEF"
os.remove("/Users/user7/Downloads/HD6_12er/Lauren & Matt Wedding/2017-09-08/LM_NEF/.DS_Store")


t = TicToc() #create instance of class
t.tic()
LM_Feature_32bins = NEFFolder2Feature32bins(LM_NEF)
t.toc()

LM_Feature_32bins.head()

# Rename the columns names
channels = ['R', 'G', 'B']
new_columns = []
for ch in channels:
    k = 0
    for i in range(32):
        col_name = ch + str(k) + "_" + str(k+7)
        new_columns.append(col_name)
        k = k+8
LM_Feature_32bins.columns = new_columns
LM_Feature_32bins.head()

## Append the features
LM_IP_Label_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDataPrepBins16/LM_Label_IndoorPerson.csv"
LM_IP_Labels = pd.read_csv(LM_IP_Label_path, index_col = 0)

LM_IP_Labels.head()

LM_IP_Features_Labels_32bins = LM_Feature_32bins.join(LM_IP_Labels, how = "inner")

LM_IP_Features_Labels_32bins.head()

def DataFrame2CSV(df, csv_folder_path, csv_file_name):
    
    csv_file_path = os.path.join(csv_folder_path, csv_file_name)
    df.to_csv(csv_file_path)

csv_folder_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDataPrepBins16"
csv_file_name_IP = "LM_IP_Features_Labels_32bins.csv"
DataFrame2CSV(LM_IP_Features_Labels_32bins, csv_folder_path, csv_file_name_IP)

LM_IP_Features_Labels_32bins.shape


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

def hist_plot_dataset(dataset, col):
    ## Distribution plot of values
    sns.distplot(dataset[col])
    plt.title('Histogram of ' + col)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.show()

for col in LM_IP_Features_Labels_32bins.columns[:96]:
    hist_plot_dataset(LM_IP_Features_Labels_32bins, col)

# Try fit with linear regression model
# set Feature array and labels array
Features = np.array(LM_IP_Features_Labels_32bins)[:,:96]
Labels_exposure = np.array(LM_IP_Features_Labels_32bins)[:,96]
Labels_contrast = np.array(LM_IP_Features_Labels_32bins)[:,97]
Labels_highlights = np.array(LM_IP_Features_Labels_32bins)[:,98]
Labels_shadows = np.array(LM_IP_Features_Labels_32bins)[:,99]

# print the shape of the features and label array
print(Features.shape)
print(Labels_exposure.shape)

# Split train and test set
X_train, X_test, y_train, y_test = train_test_split(Features, Labels_exposure, test_size=0.3, random_state=1122)

## Normalize the features
scaler = preprocessing.StandardScaler().fit(X_train) # use MinMaxScaler() because all datapoints are between 0-1
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



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


# Set checkpoints
checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]


# Train the model 
NN_model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)

# Predict the test set to evaluate the model
y_score = NN_model.predict(X_test)

# Print the evluation Results and metrics
print_metrics(y_test, y_score)
hist_resids(y_test, y_score)  
resid_qq(y_test, y_score) 
resid_plot(y_test, y_score) 

"""
NN network seems not working
Mean Square Error      = 0.120918929899079
Root Mean Square Error = 0.34773399301632707
Mean Absolute Error    = 0.2602060150626031
Median Absolute Error  = 0.2075884461402893
R^2                    = 0.1943690898319963
"""



## Try feature selection
### Now try model selection
## Define the variance threhold and fit the threshold to the feature array. 

## Define the variance threhold and fit the threshold to the feature array. 
sel = fs.VarianceThreshold(threshold=0.16)
Features_reduced = sel.fit_transform(X_train)
# All features are good

## Print the support and shape for the transformed features
print(sel.get_support())
print(Features_reduced.shape)

# select k best feature
## Reshape the Label array
Labels = y_train
Labels = Labels.reshape(Labels.shape[0],)
## Set folds for nested cross validation
nr.seed(988)
feature_folds = ms.KFold(n_splits=10, shuffle = True)

## Define the model
linear_mod = linear_model.LinearRegression()

## Perform feature selection by CV with high variance features only
nr.seed(6677)
selector = fs.RFECV(estimator = linear_mod, cv = feature_folds) # scoring = sklearn.metrics.r2_score
selector = selector.fit(Features_reduced, Labels)
selector.support_ 
selector.ranking_

Features_reduced = selector.transform(Features_reduced)
Features_reduced.shape

# Transform X_test
X_test_reduced = selector.transform(X_test)


plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
plt.title('Mean AUC by number of features')
plt.ylabel('AUC')
plt.xlabel('Number of features')

# first linear regression model

def plot_regularization(l, train_RMSE, test_RMSE, coefs, min_idx, title):   
    plt.plot(l, test_RMSE, color = 'red', label = 'Test RMSE')
    plt.plot(l, train_RMSE, label = 'Train RMSE')    
    plt.axvline(min_idx, color = 'black', linestyle = '--')
    plt.legend()
    plt.xlabel('Regularization parameter')
    plt.ylabel('Root Mean Square Error')
    plt.title(title)
    plt.show()
    
    plt.plot(l, coefs)
    plt.axvline(min_idx, color = 'black', linestyle = '--')
    plt.title('Model coefficient values \n vs. regularizaton parameter')
    plt.xlabel('Regularization parameter')
    plt.ylabel('Model coefficient value')
    plt.show()

def test_regularization_l1(x_train, y_train, x_test, y_test, l1):
    train_RMSE = []
    test_RMSE = []
    coefs = []
    for reg in l1:
        lin_mod = linear_model.Lasso(alpha = reg)
        lin_mod.fit(x_train, y_train)
        coefs.append(lin_mod.coef_)
        y_score_train = lin_mod.predict(x_train)
        train_RMSE.append(sklm.mean_squared_error(y_train, y_score_train))
        y_score = lin_mod.predict(x_test)
        test_RMSE.append(sklm.mean_squared_error(y_test, y_score))
    min_idx = np.argmin(test_RMSE)
    min_l1 = l1[min_idx]
    min_RMSE = test_RMSE[min_idx]
    
    title = 'Train and test root mean square error \n vs. regularization parameter'
    plot_regularization(l1, train_RMSE, test_RMSE, coefs, min_l1, title)
    return min_l1, min_RMSE

l1 = [x/5000 for x in range(1,101)]
out_l1 = test_regularization_l1(Features_reduced, Labels, X_test_reduced, y_test, l1)
print(out_l1)

lin_mod_l1 = linear_model.Lasso(alpha = out_l1[0])
lin_mod_l1.fit(Features_reduced, Labels)
y_score_l1 = lin_mod_l1.predict(X_test_reduced)

print_metrics(y_test, y_score_l1)
hist_resids(y_test, y_score_l1)  
resid_qq(y_test, y_score_l1) 
resid_plot(y_test, y_score_l1) 

### Linear Regression after feature selection works better
"""
Mean Square Error      = 0.09730644984450643
Root Mean Square Error = 0.3119398176644117
Mean Absolute Error    = 0.24740156079908904
Median Absolute Error  = 0.2161468255556137
R^2                    = 0.3516889057910525
"""





## Features
FixedSkew_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDataPrepBins16/IP_dataset_FixedSkew.csv"
IP_dataset_FixedSkew = pd.read_csv(FixedSkew_path,  index_col = 0)

## Test the result with NN models
IP_dataset_FixedSkew.shape
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

hist_plot(Labels_exposure, "exposure") 
np.min(Labels_exposure)
np.max(Labels_exposure)







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




## Try the model with 32 features, all together
# Save to DF based on groups
# Rename the columns names
channels = ['R', 'G', 'B']
new_columns = []
for ch in channels:
    k = 0
    for i in range(32):
        col_name = ch + str(k) + "_" + str(k+7)
        new_columns.append(col_name)
        k = k+8
LM_Feature_32bins.columns = new_columns
LM_Feature_32bins.head()

## Append the features
LM_IP_Label_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDataPrepBins16/LM_Label_IndoorPerson.csv"
LM_IP_Labels = pd.read_csv(LM_IP_Label_path, index_col = 0)

LM_IP_Labels.head()

LM_IP_Features_Labels_32bins = LM_Feature_32bins.join(LM_IP_Labels, how = "inner")

LM_IP_Features_Labels_32bins.head()

def DataFrame2CSV(df, csv_folder_path, csv_file_name):
    
    csv_file_path = os.path.join(csv_folder_path, csv_file_name)
    df.to_csv(csv_file_path)

csv_folder_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDataPrepBins16"
csv_file_name_IP = "LM_IP_Features_Labels_32bins.csv"
DataFrame2CSV(LM_IP_Features_Labels_32bins, csv_folder_path, csv_file_name_IP)

LM_IP_Features_Labels_32bins.shape

## Define a function to read in a folder path and add all csv into one df
def Addcsv(csv_folder_path, target_folder_path, new_csv_name):

        targetDF = pd.DataFrame()
        # go through all csv files in folder
        files = sorted(os.listdir(csv_folder_path))

        for file_idx in range(len(files)):
                file_name = files[file_idx]
                csv_file_path = os.path.join(csv_folder_path,file_name)
                
                dataset = pd.read_csv(csv_file_path, index_col = 0) 
                targetDF = targetDF.append(dataset)

        DataFrame2CSV(targetDF, target_folder_path, new_csv_name)

        return targetDF

# define a function to join 2 csv file in a folder into one df
def Joincsv(csv_folder_path, target_folder_path, new_csv_name):
        targetDF = pd.DataFrame()
        csv_file_path_list = []
        files = sorted(os.listdir(csv_folder_path))
        for file_idx in range(len(files)):
                file_name = files[file_idx]
                csv_file_path = os.path.join(csv_folder_path,file_name)
                csv_file_path_list.append(csv_file_path)
        
        dataset1 = pd.read_csv(csv_file_path_list[0], index_col = 0) 
        dataset2 = pd.read_csv(csv_file_path_list[1], index_col = 0) 

        targetDF = dataset1.join(dataset2, how = 'inner')

        DataFrame2CSV(targetDF, target_folder_path, new_csv_name)

        return targetDF

## Start with ED folder:save the Features_32bins.csv and Labels csv in one folder
# join the OP labels with features to create the OP_Features_Label_DF

csv_folder_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/ED_32bins_OP"
target_folder_path= "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/ED_32bins"
new_csv_name = "ED_OP_Features_Labels_32bins.csv"

ED_OP_Features_Labels_32bins_DF = Joincsv(csv_folder_path, target_folder_path, new_csv_name)


# Join the IT labels with features 
csv_folder_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/ED_32bins_IT"
target_folder_path= "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/ED_32bins"
new_csv_name = "ED_IT_Features_Labels_32bins.csv"

ED_IT_Features_Labels_32bins_DF = Joincsv(csv_folder_path, target_folder_path, new_csv_name)


# join the OT labels with features
csv_folder_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/ED_32bins_OT"
target_folder_path= "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/ED_32bins"
new_csv_name = "ED_OT_Features_Labels_32bins.csv"

ED_OT_Features_Labels_32bins_DF = Joincsv(csv_folder_path, target_folder_path, new_csv_name)

## now the four categories dataframe are save in one folder
# add the four dataframe into one csv files
# join the OT labels with features
csv_folder_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/ED_32bins"
target_folder_path= csv_folder_path
new_csv_name = "ED_Features_Labels_32bins.csv"

ED_Features_Labels_DF = Addcsv(csv_folder_path, target_folder_path, new_csv_name)
ED_Features_Labels_DF.head()
csv = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/ED_Feature.csv"
ed = pd.read_csv(csv, index_col = 0)
ed.shape


### Now work with JY data
# join the OP labels with features to create the OP_Features_Label_DF

csv_folder_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/JY_OP_32bins"
target_folder_path= "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/JY_32bins"
new_csv_name = "JY_OP_Features_Labels_32bins.csv"

JY_OP_Features_Labels_32bins_DF = Joincsv(csv_folder_path, target_folder_path, new_csv_name)


# Join the IT labels with features 
csv_folder_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/JY_IT_32bins"
target_folder_path= "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/JY_32bins"
new_csv_name = "JY_IT_Features_Labels_32bins.csv"

JY_IT_Features_Labels_32bins_DF = Joincsv(csv_folder_path, target_folder_path, new_csv_name)


# join the OT labels with features
csv_folder_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/JY_OT_32bins"
target_folder_path= "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/JY_32bins"
new_csv_name = "JY_OT_Features_Labels_32bins.csv"

JY_OT_Features_Labels_32bins_DF = Joincsv(csv_folder_path, target_folder_path, new_csv_name)

## now the four categories dataframe are save in one folder
# add the four dataframe into one csv files
csv_folder_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/JY_32bins"
target_folder_path= csv_folder_path
new_csv_name = "JY_Features_Labels_32bins.csv"
JY_Features_Labels_DF = Addcsv(csv_folder_path, target_folder_path, new_csv_name)
JY_Features_Labels_DF.head()

## Work with LC folder
# join the IP labels with features to create the OP_Features_Label_DF
csv_folder_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/LC_IP_32bins"
target_folder_path= "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/LC_32bins"
new_csv_name = "LC_IP_Features_Labels_32bins.csv"
LC_IP_Features_Labels_32bins_DF = Joincsv(csv_folder_path, target_folder_path, new_csv_name)
LC_IP_Features_Labels_32bins_DF.head()




# join the OP labels with features to create the OP_Features_Label_DF
csv_folder_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/LC_OP_32bins"
target_folder_path= "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/LC_32bins"
new_csv_name = "LC_OP_Features_Labels_32bins.csv"

LC_OP_Features_Labels_32bins_DF = Joincsv(csv_folder_path, target_folder_path, new_csv_name)
LC_OP_Features_Labels_32bins_DF.head()


LC_Features_32bins_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/LC_32bins/LC_Feature_32bins.csv"
LC_Features_32bins = pd.read_csv(LC_Features_32bins_path, index_col = 0)
LC_Features_32bins.shape
LC_Features_32bins.head()
new_columns
LC_Features_32bins.columns = new_columns
LC_Features_32bins.head()


def DataFrame2CSV(df, csv_folder_path, csv_file_name):
    
    csv_file_path = os.path.join(csv_folder_path, csv_file_name)
    df.to_csv(csv_file_path)

csv_folder_path ="/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/LC_32bins"
csv_file_name = "LC_Features_32bins.csv"
DataFrame2CSV(LC_Features_32bins, csv_folder_path, csv_file_name)
dataset = pd.read_csv("/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/LC_32bins/LC_Features_32bins.csv", index_col = 0)
dataset.head()

csv_IP = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/LC_32bins/LC_IP_Features_Labels_32bins.csv"
LC_IP_Features = pd.read_csv(csv_IP, index_col = 0)
LC_IP_Features.shape
LC_IP_Features.head()

LC_IP_Label_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/LC_32bins/LC_Label_IndoorPerson.csv"
LC_IP_Label_Df = pd.read_csv(LC_IP_Label_path, index_col = 0)
LC_IP_Label_Df.shape


LC_OP_Label_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/LC_32bins/LC_Label_OutdoorPerson.csv"
LC_OP_Label_Df = pd.read_csv(LC_OP_Label_path, index_col = 0)
LC_OP_Label_Df.shape
# fix the OP_Features_Labels_32bins dataframe
LC_OP_Features_Labels_32bins_DF = LC_Features_32bins.join(LC_OP_Label_Df, how = "inner")
LC_OP_Features_Labels_32bins_DF.shape
LC_OP_Features_Labels_32bins_DF.head()
# Save LC_OP_Features_Labels_32bins_DF as csv
csv_folder_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/LC_32bins"
csv_file_name = "LC_OP_Features_Labels_32bins.csv"
DataFrame2CSV(LC_OP_Features_Labels_32bins_DF, csv_folder_path,csv_file_name )

## conbine the LC IP_OP_IT_OT csv into one


LC_IT_Label_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/LC_32bins/LC_Label_IndoorThings.csv"
LC_IT_Label_Df = pd.read_csv(LC_IT_Label_path, index_col = 0)
LC_IT_Label_Df.shape

LC_OT_Label_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/LC_32bins/LC_Label_OutdoorThings.csv"
LC_OT_Label_Df = pd.read_csv(LC_OT_Label_path, index_col = 0)
LC_OT_Label_Df.shape

# join LC_IP features and labels
LC_IP_Features_Labels_32bins_DF.head()
LC_IP_Features_Labels_32bins_DF.shape
LC_IP_Features_Labels_32bins_DF.columns = full_cols

## OP has a problems
LC_OP_Features_Labels_32bins_DF.head()
LC_OP_Features_Labels_32bins_DF.shape

LC_OT_Features_Labels_32bins_DF.head()
LC_OT_Features_Labels_32bins_DF.shape
LC_OT_Features_Labels_32bins_DF.columns = full_cols

LC_IT_Features_Labels_32bins_DF.head()
LC_IT_Features_Labels_32bins_DF.shape
LC_IT_Features_Labels_32bins_DF.columns = full_cols

LC_Features_Labels_32bins = LC_IP_Features_Labels_32bins_DF.append(LC_OP_Features_Labels_32bins_DF)
LC_Features_Labels_32bins.shape

LC_Features_Labels_32bins = LC_Features_Labels_32bins.append(LC_OT_Features_Labels_32bins_DF)
LC_Features_Labels_32bins.shape

LC_Features_Labels_32bins = LC_Features_Labels_32bins.append(LC_IT_Features_Labels_32bins_DF)
LC_Features_Labels_32bins.shape

csv_folder_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/LC_32bins"
new_csv_name = "LC_Features_Labels_32bins.csv"
DataFrame2CSV(LC_Features_Labels_32bins,csv_folder_path, new_csv_name)

LC_Features_Labels_32bins.head()

## now the four categories dataframe are save in one folder
# add the four dataframe into one csv files
csv_folder_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/LC"
target_folder_path= csv_folder_path
new_csv_name = "LC_Features_Labels_32bins.csv"
LC_Features_Labels_32bins = Addcsv(csv_folder_path, target_folder_path, new_csv_name)

LC_Features_Labels_32bins.head()
LC_Features_Labels_32bins.shape




## Work with LM folder
# join the OP labels with features to create the OP_Features_Label_DF
csv_folder_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/LM_OP_32bins"
target_folder_path= "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/LM_32bins"
new_csv_name = "LM_OP_Features_Labels_32bins.csv"

LM_OP_Features_Labels_32bins_DF = Joincsv(csv_folder_path, target_folder_path, new_csv_name)


# Join the IT labels with features 
csv_folder_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/LM_IT_32bins"
target_folder_path= "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/LM_32bins"
new_csv_name = "LM_IT_Features_Labels_32bins.csv"

LM_IT_Features_Labels_32bins_DF = Joincsv(csv_folder_path, target_folder_path, new_csv_name)


# join the OT labels with features
csv_folder_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/LM_OT_32bins"
target_folder_path= "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/LM_32bins"
new_csv_name = "LM_OT_Features_Labels_32bins.csv"

LM_OT_Features_Labels_32bins_DF = Joincsv(csv_folder_path, target_folder_path, new_csv_name)

## now the four categories dataframe are save in one folder
# add the four dataframe into one csv files
csv_folder_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/LM_32bins"
target_folder_path= csv_folder_path
new_csv_name = "LM_Features_Labels_32bins.csv"
LM_Features_Labels_DF = Addcsv(csv_folder_path, target_folder_path, new_csv_name)

LM_Features_Labels_DF.head()


## work with KJ folder
## now the four categories dataframe are save in one folder
# add the four dataframe into one csv files
csv_folder_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/KJ_32bins"
target_folder_path= csv_folder_path
new_csv_name = "KJ_Features_Labels_32bins.csv"
KJ_Features_Labels_DF = Addcsv(csv_folder_path, target_folder_path, new_csv_name)
KJ_Features_Labels_DF.shape


## Add 4 wedding full dataframe to one 
csv_folder_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/ED_JY_LC_LM_32bins"
target_folder_path= csv_folder_path
new_csv_name = "ED_JY_LC_LM_Features_Labels_32bins.csv"
ED_JY_LC_LM_Features_Labels_32bins = Addcsv(csv_folder_path, target_folder_path, new_csv_name)

ED_JY_LC_LM_Features_Labels_32bins.shape

## now all the photo are save in one dataframe:ED_JY_LC_LM_Features_Labels_32bins
## Visualize the data
ED_JY_LC_LM_Features_Labels_32bins.head()


## Now the dataset is ready. 
## Data visualization
def hist_plot_dataset(dataset, col):
    ## Distribution plot of values
    sns.distplot(dataset[col])
    plt.title('Histogram of ' + col)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.show()
for col in ED_JY_LC_LM_Features_Labels_32bins.columns:
        hist_plot_dataset(ED_JY_LC_LM_Features_Labels_32bins, col)

## Add right skewed features
path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/ED_JY_LC_LM_32bins/ED_JY_LC_LM_Features_Labels_32bins.csv"
ED_JY_LC_LM_Features_Labels_32bins = pd.read_csv(path, index_col = 0)

ED_JY_LC_LM_Features_Labels_32bins.columns.tolist()
right_skewed_cols = []
nor_cols = []

ED_JY_LC_LM_Features_Labels_32bins_fixedskew = pd.DataFrame()
temp_IP_dataset = pd.DataFrame()

hist_plot(ED_JY_LC_LM_Features_Labels_32bins['R0_7'], 'R0_7')
temp_IP_dataset['power1_4_R0_7'] = np.power(ED_JY_LC_LM_Features_Labels_32bins['R0_7'], 1./4)
hist_plot(temp_IP_dataset['power1_4_R0_7'], 'power1_4_R0_7')
ED_JY_LC_LM_Features_Labels_32bins_fixedskew['power1_4_R0_7'] = temp_IP_dataset['power1_4_R0_7']
ED_JY_LC_LM_Features_Labels_32bins_fixedskew.head()

hist_plot(ED_JY_LC_LM_Features_Labels_32bins['R8_15'], 'R8_15')
temp_IP_dataset['sqrt_R8_15'] = np.sqrt(ED_JY_LC_LM_Features_Labels_32bins['R8_15'])
hist_plot(temp_IP_dataset['sqrt_R8_15'], 'sqrt_R8_15')
ED_JY_LC_LM_Features_Labels_32bins_fixedskew['sqrt_R8_15'] = temp_IP_dataset['sqrt_R8_15']
ED_JY_LC_LM_Features_Labels_32bins_fixedskew.head()

hist_plot(ED_JY_LC_LM_Features_Labels_32bins['R8_15'], 'R8_15')
temp_IP_dataset['sqrt_R8_15'] = np.sqrt(ED_JY_LC_LM_Features_Labels_32bins['R8_15'])
hist_plot(temp_IP_dataset['sqrt_R8_15'], 'sqrt_R8_15')
ED_JY_LC_LM_Features_Labels_32bins_fixedskew['sqrt_R8_15'] = temp_IP_dataset['sqrt_R8_15']
ED_JY_LC_LM_Features_Labels_32bins_fixedskew.head()


hist_plot(ED_JY_LC_LM_Features_Labels_32bins['R16_23'], 'R16_23') 
temp_IP_dataset['sqrt_R16_23'] = np.sqrt(ED_JY_LC_LM_Features_Labels_32bins['R16_23']) 
hist_plot(temp_IP_dataset['sqrt_R16_23'], 'sqrt_R16_23') 
ED_JY_LC_LM_Features_Labels_32bins_fixedskew['sqrt_R16_23'] = temp_IP_dataset['sqrt_R16_23'] 
ED_JY_LC_LM_Features_Labels_32bins_fixedskew.head() 

 
hist_plot(ED_JY_LC_LM_Features_Labels_32bins['R24_31'], 'R24_31') 
temp_IP_dataset['sqrt_R24_31'] = np.sqrt(ED_JY_LC_LM_Features_Labels_32bins['R24_31']) 
hist_plot(temp_IP_dataset['sqrt_R24_31'], 'sqrt_R24_31') 
ED_JY_LC_LM_Features_Labels_32bins_fixedskew['sqrt_R24_31'] = temp_IP_dataset['sqrt_R24_31'] 
ED_JY_LC_LM_Features_Labels_32bins_fixedskew.head() 

# col from 32_39 to 120_127 is good, so just save and join
nor_cols = ['R32_39',
 'R40_47',
 'R48_55',
 'R56_63',
 'R64_71',
 'R72_79',
 'R80_87',
 'R88_95',
 'R96_103',
 'R104_111',
 'R112_119',
 'R120_127']
for col in nor_cols:
         ED_JY_LC_LM_Features_Labels_32bins_fixedskew[col] = ED_JY_LC_LM_Features_Labels_32bins[col]

ED_JY_LC_LM_Features_Labels_32bins_fixedskew.head()

skewed_cols = ['R128_135',
 'R136_143',
 'R144_151',
 'R152_159',
 'R160_167',
 'R168_175']

new_cols = []
for col in skewed_cols:
         #hist_plot(ED_JY_LC_LM_Features_Labels_32bins[col], col)
         new_col = "sqrt_" + col
         new_cols.append(new_col)
         #temp_IP_dataset[new_col] = np.sqrt(ED_JY_LC_LM_Features_Labels_32bins[col])
         #hist_plot(temp_IP_dataset[new_col], new_col)
ED_JY_LC_LM_Features_Labels_32bins_fixedskew[new_cols] = np.sqrt(ED_JY_LC_LM_Features_Labels_32bins[skewed_cols])

ED_JY_LC_LM_Features_Labels_32bins_fixedskew.head()



skewed_cols =['R176_183',
 'R184_191',
 'R192_199',
 'R200_207',
 'R208_215',
 'R216_223',
 'R224_231',
 'R232_239',
 'R240_247',
 'R248_255',
 'G0_7']

new_cols = []
for col in skewed_cols:
         hist_plot(ED_JY_LC_LM_Features_Labels_32bins[col], col)
         new_col = "power1_4_" + col
         new_cols.append(new_col)
         temp_IP_dataset[new_col] = np.power(ED_JY_LC_LM_Features_Labels_32bins[col], 1./4)
         hist_plot(temp_IP_dataset[new_col], new_col)

new_cols = []
for col in skewed_cols:
         #hist_plot(ED_JY_LC_LM_Features_Labels_32bins[col], col)
         new_col = "power1_4_" + col
         new_cols.append(new_col)
new_cols


ED_JY_LC_LM_Features_Labels_32bins_fixedskew[new_cols] = np.power(ED_JY_LC_LM_Features_Labels_32bins[skewed_cols], 1./4)
ED_JY_LC_LM_Features_Labels_32bins_fixedskew.head()


nor_cols =['G8_15',
 'G16_23',
 'G24_31',
 'G32_39',
 'G40_47',
 'G48_55',
 'G56_63',
 'G64_71',
 'G72_79',
 'G80_87']
ED_JY_LC_LM_Features_Labels_32bins_fixedskew[nor_cols] = ED_JY_LC_LM_Features_Labels_32bins[nor_cols]

ED_JY_LC_LM_Features_Labels_32bins_fixedskew.head()


skewed_cols = ['G88_95',
 'G96_103',
 'G104_111',
 'G112_119',
 'G120_127',
 'G128_135']

new_cols = []
for col in skewed_cols:
         hist_plot(ED_JY_LC_LM_Features_Labels_32bins[col], col)
         new_col = "sqrt_" + col
         new_cols.append(new_col)
         temp_IP_dataset[new_col] = np.sqrt(ED_JY_LC_LM_Features_Labels_32bins[col])
         hist_plot(temp_IP_dataset[new_col], new_col)

new_cols = []
for col in skewed_cols:
         #hist_plot(ED_JY_LC_LM_Features_Labels_32bins[col], col)
         new_col = "sqrt_" + col
         new_cols.append(new_col)
new_cols


ED_JY_LC_LM_Features_Labels_32bins_fixedskew[new_cols] = np.sqrt(ED_JY_LC_LM_Features_Labels_32bins[skewed_cols])
ED_JY_LC_LM_Features_Labels_32bins_fixedskew.head()



skewed_cols = ['G136_143',
 'G144_151',
 'G152_159',
 'G160_167',
 'G168_175',
 'G176_183',
 'G184_191',
 'G192_199',
 'G200_207',
 'G208_215',
 'G216_223',
 'G224_231',
 'G232_239',
 'G240_247',
 'G248_255',
 'B0_7']

new_cols = []
for col in skewed_cols:
         hist_plot(ED_JY_LC_LM_Features_Labels_32bins[col], col)
         new_col = "power1_4_" + col
         new_cols.append(new_col)
         temp_IP_dataset[new_col] = np.power(ED_JY_LC_LM_Features_Labels_32bins[col], 1./4)
         hist_plot(temp_IP_dataset[new_col], new_col)

new_cols = []
for col in skewed_cols:
         #hist_plot(ED_JY_LC_LM_Features_Labels_32bins[col], col)
         new_col = "power1_4_" + col
         new_cols.append(new_col)
new_cols


ED_JY_LC_LM_Features_Labels_32bins_fixedskew[new_cols] = np.power(ED_JY_LC_LM_Features_Labels_32bins[skewed_cols], 1./4)
ED_JY_LC_LM_Features_Labels_32bins_fixedskew.head()


nor_cols = ['B8_15',
 'B16_23',
 'B24_31',
 'B32_39',
 'B40_47']
ED_JY_LC_LM_Features_Labels_32bins_fixedskew[nor_cols] = ED_JY_LC_LM_Features_Labels_32bins[nor_cols]

ED_JY_LC_LM_Features_Labels_32bins_fixedskew.head()


skewed_cols =['B48_55',
 'B56_63',
 'B64_71',
 'B72_79',
 'B80_87',
 'B88_95',
 'B96_103']

new_cols = []
for col in skewed_cols:
         hist_plot(ED_JY_LC_LM_Features_Labels_32bins[col], col)
         new_col = "sqrt_" + col
         new_cols.append(new_col)
         temp_IP_dataset[new_col] = np.sqrt(ED_JY_LC_LM_Features_Labels_32bins[col])
         hist_plot(temp_IP_dataset[new_col], new_col)

new_cols = []
for col in skewed_cols:
         #hist_plot(ED_JY_LC_LM_Features_Labels_32bins[col], col)
         new_col = "sqrt_" + col
         new_cols.append(new_col)
new_cols


ED_JY_LC_LM_Features_Labels_32bins_fixedskew[new_cols] = np.sqrt(ED_JY_LC_LM_Features_Labels_32bins[skewed_cols])
ED_JY_LC_LM_Features_Labels_32bins_fixedskew.head()




skewed_cols = ['B104_111',
 'B112_119',
 'B120_127',
 'B128_135',
 'B136_143',
 'B144_151',
 'B152_159',
 'B160_167',
 'B168_175',
 'B176_183',
 'B184_191',
 'B192_199',
 'B200_207',
 'B208_215',
 'B216_223',
 'B224_231',
 'B232_239',
 'B240_247',
 'B248_255']

new_cols = []
for col in skewed_cols:
         hist_plot(ED_JY_LC_LM_Features_Labels_32bins[col], col)
         new_col = "power1_4_" + col
         new_cols.append(new_col)
         temp_IP_dataset[new_col] = np.power(ED_JY_LC_LM_Features_Labels_32bins[col], 1./4)
         hist_plot(temp_IP_dataset[new_col], new_col)

new_cols = []
for col in skewed_cols:
         #hist_plot(ED_JY_LC_LM_Features_Labels_32bins[col], col)
         new_col = "power1_4_" + col
         new_cols.append(new_col)
new_cols


ED_JY_LC_LM_Features_Labels_32bins_fixedskew[new_cols] = np.power(ED_JY_LC_LM_Features_Labels_32bins[skewed_cols], 1./4)
ED_JY_LC_LM_Features_Labels_32bins_fixedskew.head()


label_cols = ['Exposure','Contrast','Highlights', 'Shadows', 'Temperature']

ED_JY_LC_LM_Features_Labels_32bins_fixedskew[label_cols] = ED_JY_LC_LM_Features_Labels_32bins[label_cols]
ED_JY_LC_LM_Features_Labels_32bins_fixedskew.head()
ED_JY_LC_LM_Features_Labels_32bins_fixedskew.shape
path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/ED_JY_LC_LM_32bins"
name = "ED_JY_LC_LM_Features_Labels_32bins_fixedskew.csv"
DataFrame2CSV(ED_JY_LC_LM_Features_Labels_32bins_fixedskew,path, name)

for col in ED_JY_LC_LM_Features_Labels_32bins_fixedskew.columns:
        hist_plot_dataset(ED_JY_LC_LM_Features_Labels_32bins_fixedskew, col)

### Now the features are more normal distributed.
## Now test with linear regression model and feature selection
import sklearn
import sklearn.metrics as sklm
import sklearn.model_selection as ms
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Dense, Flatten
from keras.models import Sequential
from matplotlib import pyplot as plt
from pytictoc import TicToc
## Define the linear regression model
from sklearn import feature_selection as fs
from sklearn import gaussian_process, linear_model
from sklearn import metrics as sklm
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.neural_network import MLPRegressor


## 
# set Feature array and labels array
Features = np.array(ED_JY_LC_LM_Features_Labels_32bins_fixedskew)[:,:96]
Labels_exposure = np.array(ED_JY_LC_LM_Features_Labels_32bins_fixedskew)[:,96]
Labels_contrast = np.array(ED_JY_LC_LM_Features_Labels_32bins_fixedskew)[:,97]
Labels_highlights = np.array(ED_JY_LC_LM_Features_Labels_32bins_fixedskew)[:,98]
Labels_shadows = np.array(ED_JY_LC_LM_Features_Labels_32bins_fixedskew)[:,99]

# print the shape of the features and label array
print(Features.shape)
print(Labels_exposure.shape)

# Split train and test set
X_train, X_test, y_train, y_test = train_test_split(Features, Labels_exposure, test_size=0.3, random_state=1122)

## Normalize the features
scaler = preprocessing.StandardScaler().fit(X_train) # use MinMaxScaler() because all datapoints are between 0-1
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


## LInear regression, no feature selection, regularization
linear_mod = linear_model.LinearRegression()
linear_mod.fit(X_train, y_train)

# predict the y_score
y_score = linear_mod.predict(X_test)

# Print the evluation Results and metrics
print_metrics(y_test, y_score)
hist_resids(y_test, y_score)  
resid_qq(y_test, y_score) 
resid_plot(y_test, y_score) 

"""
Result ED_JY_LC_LM_Features_Labels_32bins_fixedskew,  linearRegression(), no feature selection, regularization
Mean Square Error      = 0.1997249263801396
Root Mean Square Error = 0.4469059480250175
Mean Absolute Error    = 0.34720962236450925
Median Absolute Error  = 0.27925653299457437
"""


## Try with l1 regularization
linear_mod = linear_model.Lasso()
linear_mod.fit(X_train, y_train)

# predict the y_score
y_score = linear_mod.predict(X_test)

# Print the evluation Results and metrics
print_metrics(y_test, y_score)
hist_resids(y_test, y_score)  
resid_qq(y_test, y_score) 
resid_plot(y_test, y_score) 

"""
Result ED_JY_LC_LM_Features_Labels_32bins_fixedskew,  Lasso(), no feature selection, regularization


print_metrics(y_test, y_score)...
Mean Square Error      = 0.26320353113667877
Root Mean Square Error = 0.5130336549746798
Mean Absolute Error    = 0.3996290333505597
Median Absolute Error  = 0.33020228215767633
R^2                    = -0.0062668849504399216
"""



## Feature selection


## Define the variance threhold and fit the threshold to the feature array. 

sel = fs.VarianceThreshold(threshold=0.16)
Features_reduced = sel.fit_transform(X_train)

# All features are good
## Print the support and shape for the transformed features

print(sel.get_support())
print(Features_reduced.shape)


# select k best feature
## Reshape the Label array
Labels = y_train
Labels = Labels.reshape(Labels.shape[0],)
## Set folds for nested cross validation
nr.seed(988)
feature_folds = ms.KFold(n_splits=10, shuffle = True)

## Define the model
linear_mod = linear_model.LinearRegression()

## Perform feature selection by CV with high variance features only
nr.seed(6677)
selector = fs.RFECV(estimator = linear_mod, cv = feature_folds) # scoring = sklearn.metrics.r2_score
selector = selector.fit(Features_reduced, Labels)
selector.support_ 
selector.ranking_

Features_reduced = selector.transform(Features_reduced)
Features_reduced.shape

# Transform X_test
X_test_reduced = selector.transform(X_test)
X_test_reduced.shape

plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
plt.title('Mean AUC by number of features')
plt.ylabel('AUC')
plt.xlabel('Number of features')

# first linear regression model

def plot_regularization(l, train_RMSE, test_RMSE, coefs, min_idx, title):   
    plt.plot(l, test_RMSE, color = 'red', label = 'Test RMSE')
    plt.plot(l, train_RMSE, label = 'Train RMSE')    
    plt.axvline(min_idx, color = 'black', linestyle = '--')
    plt.legend()
    plt.xlabel('Regularization parameter')
    plt.ylabel('Root Mean Square Error')
    plt.title(title)
    plt.show()
    
    plt.plot(l, coefs)
    plt.axvline(min_idx, color = 'black', linestyle = '--')
    plt.title('Model coefficient values \n vs. regularizaton parameter')
    plt.xlabel('Regularization parameter')
    plt.ylabel('Model coefficient value')
    plt.show()

def test_regularization_l1(x_train, y_train, x_test, y_test, l1):
    train_RMSE = []
    test_RMSE = []
    coefs = []
    for reg in l1:
        lin_mod = linear_model.Lasso(alpha = reg)
        lin_mod.fit(x_train, y_train)
        coefs.append(lin_mod.coef_)
        y_score_train = lin_mod.predict(x_train)
        train_RMSE.append(sklm.mean_squared_error(y_train, y_score_train))
        y_score = lin_mod.predict(x_test)
        test_RMSE.append(sklm.mean_squared_error(y_test, y_score))
    min_idx = np.argmin(test_RMSE)
    min_l1 = l1[min_idx]
    min_RMSE = test_RMSE[min_idx]
    
    title = 'Train and test root mean square error \n vs. regularization parameter'
    plot_regularization(l1, train_RMSE, test_RMSE, coefs, min_l1, title)
    return min_l1, min_RMSE

l1 = [x/5000 for x in range(1,101)]
out_l1 = test_regularization_l1(Features_reduced, Labels, X_test_reduced, y_test, l1)
print(out_l1)

lin_mod_l1 = linear_model.Lasso(alpha = out_l1[0])
lin_mod_l1.fit(Features_reduced, Labels)
y_score_l1 = lin_mod_l1.predict(X_test_reduced)

print_metrics(y_test, y_score_l1)
hist_resids(y_test, y_score_l1)  
resid_qq(y_test, y_score_l1) 
resid_plot(y_test, y_score_l1) 


## with not reduced features, use NN
# Split train and test set
X_train, X_test, y_train, y_test = train_test_split(Features, Labels_exposure, test_size=0.3, random_state=1122)
X_train.shape
X_test.shape
y_train.shape
y_test.shape
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


# Set checkpoints
checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]


# Train the model 
NN_model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)

# Predict the test set to evaluate the model
y_score = NN_model.predict(X_test)

# Print the evluation Results and metrics
print_metrics(y_test, y_score)
hist_resids(y_test, y_score)  
resid_qq(y_test, y_score) 
resid_plot(y_test, y_score) 
"""
Result ED_JY_LC_LM_Features_Labels_32bins_fixedskew,  NN_6layer, no feature selection, regularization

Mean Square Error      = 0.15963152853986745
Root Mean Square Error = 0.39953914519089045
Mean Absolute Error    = 0.2741413555754644
Median Absolute Error  = 0.181494128704071
R^2                    = 0.3897045367515407
"""


### Try with indoorperson category only
ED_IP_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/ED_32bins/ED_IP_Features_Labels_32bins.csv"
ED_IP_Features_Labels_32bins = pd.read_csv(ED_IP_path, index_col = 0)
ED_IP_Features_Labels_32bins.head()
ED_IP_Features_Labels_32bins.shape

JY_IP_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/JY_32bins/JY_IP_Features_Labels_32bins.csv" 
JY_IP_Features_Labels_32bins = pd.read_csv(JY_IP_path, index_col = 0) 
JY_IP_Features_Labels_32bins.head() 
JY_IP_Features_Labels_32bins.shape 

LM_IP_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/LM_32bins/LM_IP_Features_Labels_32bins.csv" 
LM_IP_Features_Labels_32bins = pd.read_csv(LM_IP_path, index_col = 0) 
LM_IP_Features_Labels_32bins.head() 
LM_IP_Features_Labels_32bins.shape 

LC_IP_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/LC_32bins/LC_IP_Features_Labels_32bins.csv" 
LC_IP_Features_Labels_32bins = pd.read_csv(LC_IP_path, index_col = 0) 
LC_IP_Features_Labels_32bins.head() 
LC_IP_Features_Labels_32bins.shape 

### Append the four IP dataframe to one
ED_JY_LC_LM_IP_Features_Labels_32bins = ED_IP_Features_Labels_32bins.append(JY_IP_Features_Labels_32bins)
ED_JY_LC_LM_IP_Features_Labels_32bins.shape
ED_JY_LC_LM_IP_Features_Labels_32bins = ED_JY_LC_LM_IP_Features_Labels_32bins.append(LM_IP_Features_Labels_32bins)
ED_JY_LC_LM_IP_Features_Labels_32bins.shape
ED_JY_LC_LM_IP_Features_Labels_32bins = ED_JY_LC_LM_IP_Features_Labels_32bins.append(LC_IP_Features_Labels_32bins)
ED_JY_LC_LM_IP_Features_Labels_32bins.shape

def DataFrame2CSV(df, csv_folder_path, csv_file_name):
    
    csv_file_path = os.path.join(csv_folder_path, csv_file_name)
    df.to_csv(csv_file_path)
csv_folder_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/ED_JY_LC_LM_32bins"
csv_file_name = "ED_JY_LC_LM_IP_Features_Labels_32bins.csv"
DataFrame2CSV(ED_JY_LC_LM_IP_Features_Labels_32bins, csv_folder_path, csv_file_name)


## Now the four wedding dataframe IP is ready:ED_JY_LC_LM_IP_Features_Labels_32bins
## Test with NN directly, ED_JY_LC_LM_IP_Features_Labels_32bins, no skew fixed
# set Feature array and labels array
Features = np.array(ED_JY_LC_LM_IP_Features_Labels_32bins)[:,:96]
Labels_exposure = np.array(ED_JY_LC_LM_IP_Features_Labels_32bins)[:,96]
Labels_contrast = np.array(ED_JY_LC_LM_IP_Features_Labels_32bins)[:,97]
Labels_highlights = np.array(ED_JY_LC_LM_IP_Features_Labels_32bins)[:,98]
Labels_shadows = np.array(ED_JY_LC_LM_IP_Features_Labels_32bins)[:,99]

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


# Set checkpoints
checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]


# Train the model 
NN_model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)

# Predict the test set to evaluate the model
y_score = NN_model.predict(X_test)

# Print the evluation Results and metrics
print_metrics(y_test, y_score)
hist_resids(y_test, y_score)  
resid_qq(y_test, y_score) 
resid_plot(y_test, y_score) 

"""
 NN directly, ED_JY_LC_LM_IP_Features_Labels_32bins, no skew fixed

Mean Square Error      = 0.1585571432711549
Root Mean Square Error = 0.3981923445662346
Mean Absolute Error    = 0.27471296782634275
Median Absolute Error  = 0.1795226514339447
R^2                    = 0.3563591675260238
"""



## Try dataset with fixed skewness
#ED_JY_LC_LM_Features_Labels_32bins_fixedskew dataframe is full dataframe after skew correct
## join seperately with IP_labels
for col in ED_JY_LC_LM_IP_Features_Labels_32bins.columns:
        hist_plot_dataset(ED_JY_LC_LM_IP_Features_Labels_32bins, col)
ED_JY_LC_LM_IP_Features_Labels_32bins.columns.tolist()

temp_IP_dataset = pd.DataFrame()
ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew = pd.DataFrame()

skewed1_power1_4 =['R0_7']

new_cols = []
for col in skewed1_power1_4:
         hist_plot(ED_JY_LC_LM_IP_Features_Labels_32bins[col], col)
         new_col = "power1_4_" + col
         new_cols.append(new_col)
         temp_IP_dataset[new_col] = np.power(ED_JY_LC_LM_IP_Features_Labels_32bins[col], 1./4)
         hist_plot(temp_IP_dataset[new_col], new_col)


ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew[new_cols] = np.power(ED_JY_LC_LM_IP_Features_Labels_32bins[skewed1_power1_4], 1./4)
ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew.head()

skewed2_sqrt = ['R8_15']
new_cols = [] 
for col in skewed2_sqrt: 
        hist_plot(ED_JY_LC_LM_IP_Features_Labels_32bins[col], col) 
        new_col = "sqrt_" + col 
        new_cols.append(new_col) 
        temp_IP_dataset[new_col] = np.sqrt(ED_JY_LC_LM_IP_Features_Labels_32bins[col]) 
        hist_plot(temp_IP_dataset[new_col], new_col) 

ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew[new_cols] = np.sqrt(ED_JY_LC_LM_IP_Features_Labels_32bins[skewed2_sqrt]) 
ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew.head() 


nor1 =['R16_23',
 'R24_31',
 'R32_39',
 'R40_47',
 'R48_55',
 'R56_63',
 'R64_71',
 'R72_79',
 'R80_87',
 'R88_95',
 'R96_103',
 'R104_111',
 'R112_119',
 'R120_127',
 'R128_135']
ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew[nor1] = ED_JY_LC_LM_IP_Features_Labels_32bins[nor1] 
ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew.head() 

 skew3_sqrt =['R136_143',
 'R144_151',
 'R152_159',
 'R160_167',
 'R168_175']
new_cols = [] 
for col in skew3_sqrt: 
        hist_plot(ED_JY_LC_LM_IP_Features_Labels_32bins[col], col) 
        new_col = "sqrt_" + col 
        new_cols.append(new_col) 
        temp_IP_dataset[new_col] = np.sqrt(ED_JY_LC_LM_IP_Features_Labels_32bins[col]) 
        hist_plot(temp_IP_dataset[new_col], new_col) 

ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew[new_cols] = np.sqrt(ED_JY_LC_LM_IP_Features_Labels_32bins[skew3_sqrt]) 
ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew.head() 


 skew4_power1_4 =['R176_183',
 'R184_191',
 'R192_199',
 'R200_207',
 'R208_215',
 'R216_223',
 'R224_231',
 'R232_239',
 'R240_247',
 'R248_255',
 'G0_7']
new_cols = []
for col in skew4_power1_4:
         hist_plot(ED_JY_LC_LM_IP_Features_Labels_32bins[col], col)
         new_col = "power1_4_" + col
         new_cols.append(new_col)
         temp_IP_dataset[new_col] = np.power(ED_JY_LC_LM_IP_Features_Labels_32bins[col], 1./4)
         hist_plot(temp_IP_dataset[new_col], new_col)

ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew[new_cols] = np.power(ED_JY_LC_LM_IP_Features_Labels_32bins[skew4_power1_4], 1./4)
ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew.head()
## Power1_4_R248_255 seems to have a problem. May delete to test the effect. 


 nor2 = ['G8_15',
 'G16_23',
 'G24_31',
 'G32_39',
 'G40_47',
 'G48_55',
 'G56_63',
 'G64_71',
 'G72_79']
ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew[nor2] = ED_JY_LC_LM_IP_Features_Labels_32bins[nor2] 
ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew.head() 



 skewed5_sqrt =['G80_87',
 'G88_95',
 'G96_103',
 'G104_111',
 'G112_119',
 'G120_127',
 'G128_135']
new_cols = [] 
for col in skewed5_sqrt: 
        hist_plot(ED_JY_LC_LM_IP_Features_Labels_32bins[col], col) 
        new_col = "sqrt_" + col 
        new_cols.append(new_col) 
        temp_IP_dataset[new_col] = np.sqrt(ED_JY_LC_LM_IP_Features_Labels_32bins[col]) 
        hist_plot(temp_IP_dataset[new_col], new_col) 

ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew[new_cols] = np.sqrt(ED_JY_LC_LM_IP_Features_Labels_32bins[skewed5_sqrt]) 
ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew.head() 




 skewed6_power1_4 = [ 'G136_143',
 'G144_151',
 'G152_159',
 'G160_167',
 'G168_175',
 'G176_183',
 'G184_191',
 'G192_199',
 'G200_207',
 'G208_215',
 'G216_223',
 'G224_231',
 'G232_239',
 'G240_247',
 'G248_255',
 'B0_7']
new_cols = []
for col in skewed6_power1_4:
         hist_plot(ED_JY_LC_LM_IP_Features_Labels_32bins[col], col)
         new_col = "power1_4_" + col
         new_cols.append(new_col)
         temp_IP_dataset[new_col] = np.power(ED_JY_LC_LM_IP_Features_Labels_32bins[col], 1./4)
         hist_plot(temp_IP_dataset[new_col], new_col)

ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew[new_cols] = np.power(ED_JY_LC_LM_IP_Features_Labels_32bins[skewed6_power1_4], 1./4)
ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew.head()




 nor3 =['B8_15',
 'B16_23',
 'B24_31',
 'B32_39',
 'B40_47',
 'B48_55',
 'B56_63']
ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew[nor3] = ED_JY_LC_LM_IP_Features_Labels_32bins[nor3] 
ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew.head() 




 skewed7_sqrt =['B64_71',
 'B72_79',
 'B80_87',
 'B88_95']
new_cols = [] 
for col in skewed7_sqrt: 
        hist_plot(ED_JY_LC_LM_IP_Features_Labels_32bins[col], col) 
        new_col = "sqrt_" + col 
        new_cols.append(new_col) 
        temp_IP_dataset[new_col] = np.sqrt(ED_JY_LC_LM_IP_Features_Labels_32bins[col]) 
        hist_plot(temp_IP_dataset[new_col], new_col) 

ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew[new_cols] = np.sqrt(ED_JY_LC_LM_IP_Features_Labels_32bins[skewed7_sqrt]) 
ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew.head() 



 skewed8_power1_4 = ['B96_103',
 'B104_111',
 'B112_119',
 'B120_127',
 'B128_135',
 'B136_143',
 'B144_151',
 'B152_159',
 'B160_167',
 'B168_175',
 'B176_183',
 'B184_191',
 'B192_199',
 'B200_207',
 'B208_215',
 'B216_223',
 'B224_231',
 'B232_239',
 'B240_247',
 'B248_255']
new_cols = []
for col in skewed8_power1_4:
         hist_plot(ED_JY_LC_LM_IP_Features_Labels_32bins[col], col)
         new_col = "power1_4_" + col
         new_cols.append(new_col)
         temp_IP_dataset[new_col] = np.power(ED_JY_LC_LM_IP_Features_Labels_32bins[col], 1./4)
         hist_plot(temp_IP_dataset[new_col], new_col)

ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew[new_cols] = np.power(ED_JY_LC_LM_IP_Features_Labels_32bins[skewed8_power1_4], 1./4)
ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew.head()
## power1_4_B248_255 seems to have a problem. may delete the columns

## Add the label columns
ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew[label_cols] = ED_JY_LC_LM_IP_Features_Labels_32bins[label_cols]
ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew.shape

path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/ED_JY_LC_LM_32bins"
name = "ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew.csv"
DataFrame2CSV(ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew, path, name)



## Now test with NN, ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew
# set Feature array and labels array
Features = np.array(ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew)[:,:96]
Labels_exposure = np.array(ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew)[:,96]
Labels_contrast = np.array(ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew)[:,97]
Labels_highlights = np.array(ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew)[:,98]
Labels_shadows = np.array(ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew)[:,99]

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


# Set checkpoints
checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]


# Train the model 
NN_model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)

# Predict the test set to evaluate the model
y_score = NN_model.predict(X_test)

# Print the evluation Results and metrics
print_metrics(y_test, y_score)
hist_resids(y_test, y_score)  
resid_qq(y_test, y_score) 
resid_plot(y_test, y_score) 
"""
Result NN, ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew
Mean Square Error      = 0.14827803679193213
Root Mean Square Error = 0.38506887279022195
Mean Absolute Error    = 0.26453873407489026
Median Absolute Error  = 0.181642758846283
R^2                    = 0.3980857811296834

"""



### Try to delete the last bin of R, G, B channel
ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew_delRGBlastBin = pd.DataFrame()
new_cols = ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew.columns.tolist()
new_cols = ['power1_4_R0_7',
 'sqrt_R8_15',
 'R16_23',
 'R24_31',
 'R32_39',
 'R40_47',
 'R48_55',
 'R56_63',
 'R64_71',
 'R72_79',
 'R80_87',
 'R88_95',
 'R96_103',
 'R104_111',
 'R112_119',
 'R120_127',
 'R128_135',
 'sqrt_R136_143',
 'sqrt_R144_151',
 'sqrt_R152_159',
 'sqrt_R160_167',
 'sqrt_R168_175',
 'power1_4_R176_183',
 'power1_4_R184_191',
 'power1_4_R192_199',
 'power1_4_R200_207',
 'power1_4_R208_215',
 'power1_4_R216_223',
 'power1_4_R224_231',
 'power1_4_R232_239',
 'power1_4_R240_247',
 'power1_4_G0_7',
 'G8_15',
 'G16_23',
 'G24_31',
 'G32_39',
 'G40_47',
 'G48_55',
 'G56_63',
 'G64_71',
 'G72_79',
 'sqrt_G80_87',
 'sqrt_G88_95',
 'sqrt_G96_103',
 'sqrt_G104_111',
 'sqrt_G112_119',
 'sqrt_G120_127',
 'sqrt_G128_135',
 'power1_4_G136_143',
 'power1_4_G144_151',
 'power1_4_G152_159',
 'power1_4_G160_167',
 'power1_4_G168_175',
 'power1_4_G176_183',
 'power1_4_G184_191',
 'power1_4_G192_199',
 'power1_4_G200_207',
 'power1_4_G208_215',
 'power1_4_G216_223',
 'power1_4_G224_231',
 'power1_4_G232_239',
 'power1_4_G240_247',
 'power1_4_B0_7',
 'B8_15',
 'B16_23',
 'B24_31',
 'B32_39',
 'B40_47',
 'B48_55',
 'B56_63',
 'sqrt_B64_71',
 'sqrt_B72_79',
 'sqrt_B80_87',
 'sqrt_B88_95',
 'power1_4_B96_103',
 'power1_4_B104_111',
 'power1_4_B112_119',
 'power1_4_B120_127',
 'power1_4_B128_135',
 'power1_4_B136_143',
 'power1_4_B144_151',
 'power1_4_B152_159',
 'power1_4_B160_167',
 'power1_4_B168_175',
 'power1_4_B176_183',
 'power1_4_B184_191',
 'power1_4_B192_199',
 'power1_4_B200_207',
 'power1_4_B208_215',
 'power1_4_B216_223',
 'power1_4_B224_231',
 'power1_4_B232_239',
 'power1_4_B240_247',
 'Exposure',
 'Contrast',
 'Highlights',
 'Shadows',
 'Temperature']

ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew_delRGBlastBin[new_cols] = ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew[new_cols]


ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew_delRGBlastBin.shape

path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/ED_JY_LC_LM_32bins"
name = "ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew_delRGBlastBin.csv"
DataFrame2CSV(ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew, path, name)

## Test the NN with new dataset
## Now test with NN, ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew
# set Feature array and labels array
Features = np.array(ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew_delRGBlastBin)[:,:93]
Labels_exposure = np.array(ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew_delRGBlastBin)[:,93]
Labels_contrast = np.array(ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew_delRGBlastBin)[:,94]
Labels_highlights = np.array(ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew_delRGBlastBin)[:,95]
Labels_shadows = np.array(ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew_delRGBlastBin)[:,96]

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


# Set checkpoints
checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]


# Train the model 
NN_model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)

# Predict the test set to evaluate the model
y_score = NN_model.predict(X_test)

# Print the evluation Results and metrics
print_metrics(y_test, y_score)
hist_resids(y_test, y_score)  
resid_qq(y_test, y_score) 
resid_plot(y_test, y_score) 

"""
Result: NN, ED_JY_LC_LM_IP_Features_Labels_32bins_fixedskew_delRGBlastBin

print_metrics(y_test, y_score)...
Mean Square Error      = 0.1499483336828154
Root Mean Square Error = 0.38723162794742816
Mean Absolute Error    = 0.2663746369400256
Median Absolute Error  = 0.17560681700706482
R^2                    = 0.3913054415048185
""""


###Try Linear Regression
## Define the variance threhold and fit the threshold to the feature array. 

sel = fs.VarianceThreshold(threshold=0.16)
Features_reduced = sel.fit_transform(X_train)

# All features are good
## Print the support and shape for the transformed features

print(sel.get_support())
print(Features_reduced.shape)


# select k best feature
## Reshape the Label array
Labels = y_train
Labels = Labels.reshape(Labels.shape[0],)
## Set folds for nested cross validation
nr.seed(988)
feature_folds = ms.KFold(n_splits=10, shuffle = True)

## Define the model
linear_mod = linear_model.LinearRegression()

## Perform feature selection by CV with high variance features only
nr.seed(6677)
selector = fs.RFECV(estimator = linear_mod, cv = feature_folds) # scoring = sklearn.metrics.r2_score
selector = selector.fit(Features_reduced, Labels)
selector.support_ 
selector.ranking_

Features_reduced = selector.transform(Features_reduced)
Features_reduced.shape

# Transform X_test
X_test_reduced = selector.transform(X_test)
X_test_reduced.shape

plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
plt.title('Mean AUC by number of features')
plt.ylabel('AUC')
plt.xlabel('Number of features')

# first linear regression model

def plot_regularization(l, train_RMSE, test_RMSE, coefs, min_idx, title):   
    plt.plot(l, test_RMSE, color = 'red', label = 'Test RMSE')
    plt.plot(l, train_RMSE, label = 'Train RMSE')    
    plt.axvline(min_idx, color = 'black', linestyle = '--')
    plt.legend()
    plt.xlabel('Regularization parameter')
    plt.ylabel('Root Mean Square Error')
    plt.title(title)
    plt.show()
    
    plt.plot(l, coefs)
    plt.axvline(min_idx, color = 'black', linestyle = '--')
    plt.title('Model coefficient values \n vs. regularizaton parameter')
    plt.xlabel('Regularization parameter')
    plt.ylabel('Model coefficient value')
    plt.show()

def test_regularization_l1(x_train, y_train, x_test, y_test, l1):
    train_RMSE = []
    test_RMSE = []
    coefs = []
    for reg in l1:
        lin_mod = linear_model.Lasso(alpha = reg)
        lin_mod.fit(x_train, y_train)
        coefs.append(lin_mod.coef_)
        y_score_train = lin_mod.predict(x_train)
        train_RMSE.append(sklm.mean_squared_error(y_train, y_score_train))
        y_score = lin_mod.predict(x_test)
        test_RMSE.append(sklm.mean_squared_error(y_test, y_score))
    min_idx = np.argmin(test_RMSE)
    min_l1 = l1[min_idx]
    min_RMSE = test_RMSE[min_idx]
    
    title = 'Train and test root mean square error \n vs. regularization parameter'
    plot_regularization(l1, train_RMSE, test_RMSE, coefs, min_l1, title)
    return min_l1, min_RMSE

l1 = [x/5000 for x in range(1,101)]
out_l1 = test_regularization_l1(Features_reduced, Labels, X_test_reduced, y_test, l1)
print(out_l1)

lin_mod_l1 = linear_model.Lasso(alpha = out_l1[0])
lin_mod_l1.fit(Features_reduced, Labels)
y_score_l1 = lin_mod_l1.predict(X_test_reduced)

print_metrics(y_test, y_score_l1)
hist_resids(y_test, y_score_l1)  
resid_qq(y_test, y_score_l1) 
resid_plot(y_test, y_score_l1) 
"""
Result: linear regression, NN
Mean Square Error      = 0.18514621256439734
Root Mean Square Error = 0.4302861984358752
Mean Absolute Error    = 0.3335700722045434
Median Absolute Error  = 0.25278864178448923
R^2                    = 0.2484245116566014
"""

## So far, the best performance is with IP_dataset_FixedSkew_16bins, NN model
## Try CNN to do regression

# Try pytorch

import tensorflow as tf
x_data = np.random.rand(100).astype(np.float32)
a = tf.Variable(1.0)
b = tf.Variable(0.5)
y = a * x_data + b
print(y)

loss  = tf.reduce_mean(tf.square(y_test - y_data))
