!pip install keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import preprocessing
import sklearn.model_selection as ms
from sklearn import linear_model
import sklearn.metrics as sklm
from sklearn import feature_selection as fs
from sklearn import metrics
from sklearn.model_selection import cross_validate
import numpy.random as nr
import seaborn as sns
import scipy.stats as ss
import math
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 
from matplotlib import pyplot as plt
import seaborn as sb
from sklearn.preprocessing import MinMaxScaler

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


##Step1:  Load the dataset: new_IT_dataset_combine_ND.csv has the files with 8-bins each channel 
# new_IT_dataset_combine_ND.csv is processed to make the features normalized distribution
# Data preparation is done with 6_DataVisualizationPreparation

# set the csv file path
new_IT_dataset_combine_ND_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDataPrep/new_IT_dataset_combine_ND.csv"

# read the file as csv and use the first row as index
new_IT_dataset_combine_ND = pd.read_csv(new_IT_dataset_combine_ND_path, index_col = 0) 

# get the feature coloumns names
cols = new_IT_dataset_combine_ND.columns.tolist()[:24]
print(cols)

# review the whole dataset
new_IT_dataset_combine_ND.head()

## Set the Feature and Label arrays
#review the all the columns names
print(new_IT_dataset_combine_ND.columns)

# set Feature array and labels array
Features = np.array(new_IT_dataset_combine_ND)[:,:24]
Labels_exposure = np.array(new_IT_dataset_combine_ND)[:,24]
Labels_contrast = np.array(new_IT_dataset_combine_ND)[:,25]
Labels_highlights = np.array(new_IT_dataset_combine_ND)[:,26]
Labels_shadows = np.array(new_IT_dataset_combine_ND)[:,27]

# print the shape of the features and label array
print(Features.shape)
print(Labels_exposure.shape)

# Split train and test set
X_train, X_test, y_train, y_test = train_test_split(Features, Labels_exposure, test_size=0.3, random_state=1122)

## Normalize the features
scaler = preprocessing.MinMaxScaler().fit(X_train) # use MinMaxScaler() because all datapoints are between 0-1
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
NN_model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)

# Predict the test set to evaluate the model
y_score = NN_model.predict(X_test)

# Print the evluation Results and metrics
print_metrics(y_test, y_score)
hist_resids(y_test, y_score)  
resid_qq(y_test, y_score) 
resid_plot(y_test, y_score) 
####--->> MSE = 0.16479095306993777 bad!!!

##############################################################################
### --->>> Try the 16-bins-each channel IP dataset  ----->>>>>>>>Doesn't work!!!!
# set the csv file path
ED_JY_LM_LC_IP_Features_Labels_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDataPrep/ED_JY_LM_LC_IP_Features_Labels.csv"

# read the file as csv and use the first row as index
ED_JY_LM_LC_IP_Features_Labels = pd.read_csv(ED_JY_LM_LC_IP_Features_Labels_path, index_col = 0) 

# get the feature coloumns names
cols = ED_JY_LM_LC_IP_Features_Labels.columns.tolist()[:24]
print(cols)

# review the whole dataset
ED_JY_LM_LC_IP_Features_Labels.head()

## Set the Feature and Label arrays
#review the all the columns names
print(ED_JY_LM_LC_IP_Features_Labels.columns)

# set Feature array and labels array
Features = np.array(ED_JY_LM_LC_IP_Features_Labels)[:,:48]
Labels_exposure = np.array(ED_JY_LM_LC_IP_Features_Labels)[:,48]
Labels_contrast = np.array(ED_JY_LM_LC_IP_Features_Labels)[:,49]
Labels_highlights = np.array(ED_JY_LM_LC_IP_Features_Labels)[:,50]
Labels_shadows = np.array(ED_JY_LM_LC_IP_Features_Labels)[:,51]

# print the shape of the features and label array
print(Features.shape)
print(Labels_exposure.shape)

# Split train and test set
X_train, X_test, y_train, y_test = train_test_split(Features, Labels_exposure, test_size=0.3, random_state=1122)

## Normalize the features
scaler = preprocessing.MinMaxScaler().fit(X_train) # use MinMaxScaler() because all datapoints are between 0-1
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
NN_model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)

# Predict the test set to evaluate the model
y_score = NN_model.predict(X_test)

# Print the evluation Results and metrics
print_metrics(y_test, y_score)
hist_resids(y_test, y_score)  
resid_qq(y_test, y_score) 
resid_plot(y_test, y_score) 
#### MSE is bad !!!! 







## use original dataset, with MixMaxScaler(), made a mistake about the labels!!!!
## Load the data
ED_JY_LM_LC_IP_path = "/Users/user7/Downloads/HD6_12er/DataPreparation/ED_JY_LM_LC_IP_Features_Labels.csv"

# read the file as csv and use the first row as index
ED_JY_LM_LC_IP = pd.read_csv(ED_JY_LM_LC_IP_path, index_col = 0) 
cols = ED_JY_LM_LC_IP.columns.tolist()[:24]
cols
ED_JY_LM_LC_IP.head()


Features = np.array(ED_JY_LM_LC_IP)[:,:24]
Labels_exposure = np.array(ED_JY_LM_LC_IP)[:,24] # it is the wrong label
Labels_contrast = np.array(ED_JY_LM_LC_IP)[:,25]
Labels_highlights = np.array(ED_JY_LM_LC_IP)[:,26]
Labels_shadows = np.array(ED_JY_LM_LC_IP)[:,27]
Features.shape
Labels_exposure.shape



# Split train and test set
X_train, X_test, y_train, y_test = train_test_split(Features, Labels_exposure, test_size=0.3, random_state=1122)

## Normalize the features
scaler = preprocessing.MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


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

checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]


# Train the model
NN_model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)



# Predict
def make_submission(prediction, sub_name):
  my_submission = pd.DataFrame({'Id':pd.read_csv('test.csv').Id,'SalePrice':prediction})
  my_submission.to_csv('{}.csv'.format(sub_name),index=False)
  print('A submission file has been made')

y_score = NN_model.predict(X_test)
#make_submission(predictions[:,0],'submission(NN).csv')


print_metrics(y_test, y_score)
hist_resids(y_test, y_score)  
resid_qq(y_test, y_score) 
resid_plot(y_test, y_score) 
##### This model is actually using the R, G channel to predict the B channel













## use original dataset, with MixMaxScaler(), made a mistake about the labels!
## Load the data and predict Labels[25], which is actually the B channel
ED_JY_LM_LC_IP_path = "/Users/user7/Downloads/HD6_12er/DataPreparation/ED_JY_LM_LC_IP_Features_Labels.csv"

# read the file as csv and use the first row as index
ED_JY_LM_LC_IP = pd.read_csv(ED_JY_LM_LC_IP_path, index_col = 0) 
cols = ED_JY_LM_LC_IP.columns.tolist()[:24]
cols
ED_JY_LM_LC_IP.head()


Features = np.array(ED_JY_LM_LC_IP)[:,:24]
Labels_exposure = np.array(ED_JY_LM_LC_IP)[:,24] # it is the wrong label
Labels_contrast = np.array(ED_JY_LM_LC_IP)[:,25]
Labels_highlights = np.array(ED_JY_LM_LC_IP)[:,26]
Labels_shadows = np.array(ED_JY_LM_LC_IP)[:,27]
Features.shape
Labels_exposure.shape



# Split train and test set
X_train, X_test, y_train, y_test = train_test_split(Features, Labels_contrast, test_size=0.3, random_state=1122)

## Normalize the features
scaler = preprocessing.MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


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

checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]


# Train the model
NN_model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)



# Predict
def make_submission(prediction, sub_name):
  my_submission = pd.DataFrame({'Id':pd.read_csv('test.csv').Id,'SalePrice':prediction})
  my_submission.to_csv('{}.csv'.format(sub_name),index=False)
  print('A submission file has been made')

y_score = NN_model.predict(X_test)
#make_submission(predictions[:,0],'submission(NN).csv')


print_metrics(y_test, y_score)
hist_resids(y_test, y_score)  
resid_qq(y_test, y_score) 
resid_plot(y_test, y_score) 







## use original dataset, with MixMaxScaler(), made a mistake about the labels!
## Load the data and predict labels[32]
ED_JY_LM_LC_IP_path = "/Users/user7/Downloads/HD6_12er/DataPreparation/ED_JY_LM_LC_IP_Features_Labels.csv"

# read the file as csv and use the first row as index
ED_JY_LM_LC_IP = pd.read_csv(ED_JY_LM_LC_IP_path, index_col = 0) 
cols = ED_JY_LM_LC_IP.columns.tolist()[:24]
cols
ED_JY_LM_LC_IP.head()


Features = np.array(ED_JY_LM_LC_IP)[:,:24]
Labels_exposure = np.array(ED_JY_LM_LC_IP)[:,24] # it is the wrong label
Labels_contrast = np.array(ED_JY_LM_LC_IP)[:,25]
Labels_highlights = np.array(ED_JY_LM_LC_IP)[:,26]
Labels_shadows = np.array(ED_JY_LM_LC_IP)[:,27]
Features.shape
Labels_exposure.shape
Labels_y = np.array(ED_JY_LM_LC_IP)[:,32]


# Split train and test set
X_train, X_test, y_train, y_test = train_test_split(Features, Labels_y, test_size=0.3, random_state=1122)

## Normalize the features
scaler = preprocessing.MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


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

checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]


# Train the model
NN_model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)



# Predict
def make_submission(prediction, sub_name):
  my_submission = pd.DataFrame({'Id':pd.read_csv('test.csv').Id,'SalePrice':prediction})
  my_submission.to_csv('{}.csv'.format(sub_name),index=False)
  print('A submission file has been made')

y_score = NN_model.predict(X_test)
#make_submission(predictions[:,0],'submission(NN).csv')


print_metrics(y_test, y_score)
hist_resids(y_test, y_score)  
resid_qq(y_test, y_score) 
resid_plot(y_test, y_score) 







## use original dataset, with MixMaxScaler(), made a mistake about the labels!
## Load the data and predict labels[36]
ED_JY_LM_LC_IP_path = "/Users/user7/Downloads/HD6_12er/DataPreparation/ED_JY_LM_LC_IP_Features_Labels.csv"

# read the file as csv and use the first row as index
ED_JY_LM_LC_IP = pd.read_csv(ED_JY_LM_LC_IP_path, index_col = 0) 
cols = ED_JY_LM_LC_IP.columns.tolist()[:24]
cols
ED_JY_LM_LC_IP.head()


Features = np.array(ED_JY_LM_LC_IP)[:,:24]
Labels_exposure = np.array(ED_JY_LM_LC_IP)[:,24] # it is the wrong label
Labels_contrast = np.array(ED_JY_LM_LC_IP)[:,25]
Labels_highlights = np.array(ED_JY_LM_LC_IP)[:,26]
Labels_shadows = np.array(ED_JY_LM_LC_IP)[:,27]
Features.shape
Labels_exposure.shape
Labels_y = np.array(ED_JY_LM_LC_IP)[:,36]


# Split train and test set
X_train, X_test, y_train, y_test = train_test_split(Features, Labels_y, test_size=0.3, random_state=1122)

## Normalize the features
scaler = preprocessing.MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


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

checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]


# Train the model
NN_model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)



# Predict
def make_submission(prediction, sub_name):
  my_submission = pd.DataFrame({'Id':pd.read_csv('test.csv').Id,'SalePrice':prediction})
  my_submission.to_csv('{}.csv'.format(sub_name),index=False)
  print('A submission file has been made')

y_score = NN_model.predict(X_test)
#make_submission(predictions[:,0],'submission(NN).csv')


print_metrics(y_test, y_score)
hist_resids(y_test, y_score)  
resid_qq(y_test, y_score) 
resid_plot(y_test, y_score) 












## use original dataset, with MixMaxScaler(), made a mistake about the labels!
## Load the data and predict labels[40]
ED_JY_LM_LC_IP_path = "/Users/user7/Downloads/HD6_12er/DataPreparation/ED_JY_LM_LC_IP_Features_Labels.csv"

# read the file as csv and use the first row as index
ED_JY_LM_LC_IP = pd.read_csv(ED_JY_LM_LC_IP_path, index_col = 0) 
cols = ED_JY_LM_LC_IP.columns.tolist()[:24]
cols
ED_JY_LM_LC_IP.head()


Features = np.array(ED_JY_LM_LC_IP)[:,:24]
Labels_exposure = np.array(ED_JY_LM_LC_IP)[:,24] # it is the wrong label
Labels_contrast = np.array(ED_JY_LM_LC_IP)[:,25]
Labels_highlights = np.array(ED_JY_LM_LC_IP)[:,26]
Labels_shadows = np.array(ED_JY_LM_LC_IP)[:,27]
Features.shape
Labels_exposure.shape
Labels_y = np.array(ED_JY_LM_LC_IP)[:,40]


# Split train and test set
X_train, X_test, y_train, y_test = train_test_split(Features, Labels_y, test_size=0.3, random_state=1122)

## Normalize the features
scaler = preprocessing.MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


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

checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]


# Train the model
NN_model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)



# Predict
def make_submission(prediction, sub_name):
  my_submission = pd.DataFrame({'Id':pd.read_csv('test.csv').Id,'SalePrice':prediction})
  my_submission.to_csv('{}.csv'.format(sub_name),index=False)
  print('A submission file has been made')

y_score = NN_model.predict(X_test)
#make_submission(predictions[:,0],'submission(NN).csv')


print_metrics(y_test, y_score)
hist_resids(y_test, y_score)  
resid_qq(y_test, y_score) 
resid_plot(y_test, y_score) 








