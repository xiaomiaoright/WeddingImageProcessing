import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## Load the data
new_IT_dataset_combine_ND_path = "/Users/user7/Downloads/HD6_12er/DataPreparation/new_IT_dataset_combine_ND.csv"

# read the file as csv and use the first row as index
new_IT_dataset_combine_ND = pd.read_csv(new_IT_dataset_combine_ND_path, index_col = 0) 
cols = new_IT_dataset_combine_ND.columns.tolist()[:24]
cols
new_IT_dataset_combine_ND.head()

def plot_scatter_t(auto_prices, cols, col_y = 'price', alpha = 1.0):
    for col in cols:
        fig = plt.figure(figsize=(7,6)) # define plot area
        ax = fig.gca() # define axis   
        auto_prices.plot.scatter(x = col, y = col_y, ax = ax, alpha = alpha)
        ax.set_title('Scatter plot of ' + col_y + ' vs. ' + col) # Give the plot a main title
        ax.set_xlabel(col) # Set text for the x axis
        ax.set_ylabel(col_y)# Set text for y axis
        plt.show()

plot_scatter_t(new_IT_dataset_combine_ND, cols, 'Exposure', alpha=0.2)

plot_scatter_t(new_IT_dataset_combine_ND, cols, 'Contrast', alpha=0.2)

plot_scatter_t(new_IT_dataset_combine_ND, cols, 'Highlights', alpha=0.2)

plot_scatter_t(new_IT_dataset_combine_ND, cols, 'Shadows', alpha=0.2)


R_cols = cols[:8]
G_cols = cols[8:16]
B_cols = cols[17:]
new_cols = ['A',"B",'C','D','E','F','G','H']

updated_df = pd.DataFrame()


for i in range(8):
    updated_df[new_cols[i]] = new_IT_dataset_combine_ND[R_cols[i]]*new_IT_dataset_combine_ND[G_cols[i]]

updated_df[['Exposure','Contrast','Highlights','Shadows','Temperature']] = new_IT_dataset_combine_ND[['Exposure','Contrast','Highlights','Shadows','Temperature']]
updated_df.head()

plot_scatter_t(updated_df, new_cols, 'Exposure', alpha=0.2)

## Not helping with linear relationship, use new_IT_dataset_combine_ND for regression model

import pandas as pd
from sklearn import preprocessing
import sklearn.model_selection as ms
from sklearn import linear_model
import sklearn.metrics as sklm
from sklearn import feature_selection as fs
from sklearn import metrics
from sklearn.model_selection import cross_validate
import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import math
new_IT_dataset_combine_ND.columns
Features = np.array(new_IT_dataset_combine_ND)[:,:24]
Labels_exposure = np.array(new_IT_dataset_combine_ND)[:,24]
Labels_contrast = np.array(new_IT_dataset_combine_ND)[:,25]
Labels_highlights = np.array(new_IT_dataset_combine_ND)[:,26]
Labels_shadows = np.array(new_IT_dataset_combine_ND)[:,27]
Features.shape
Labels_exposure.shape

# Split train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Features, Labels_exposure, test_size=0.3, random_state=1122)

## Normalize the features
from sklearn.preprocessing import MinMaxScaler
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



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

print_metrics(y_test, y_score_l1)
hist_resids(y_test, y_score_l1)  
resid_qq(y_test, y_score_l1) 
resid_plot(y_test, y_score_l1) 


##### Try different model: linear_regression model
lin_mod = linear_model.LinearRegression()
lin_mod.fit(Features_reduced, Labels)
y_score = lin_mod.predict(X_test_reduced)

print_metrics(y_test, y_score)
hist_resids(y_test, y_score)  
resid_qq(y_test, y_score) 
resid_plot(y_test, y_score) 

#### Try different model: NN
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
#from statsmodels.api import datasets
from sklearn import datasets ## Get dataset from sklearn
import sklearn.model_selection as ms
import sklearn.metrics as sklm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy.random as nr

nr.seed(1115)
nn_mod = MLPRegressor(hidden_layer_sizes = (70,))
nn_mod.fit(Features_reduced, Labels)
y_score = nn_mod.predict(X_test_reduced)

print_metrics(y_test, y_score)
hist_resids(y_test, y_score)  
resid_qq(y_test, y_score) 
resid_plot(y_test, y_score) 



#### Try different model: 
from sklearn.ensemble import AdaBoostRegressor
nr.seed(444)
rf_regression = AdaBoostRegressor(learning_rate=0.5, random_state=1122)
rf_regression.fit(Features_reduced, Labels)
y_score = rf_regression.predict(X_test_reduced)

print_metrics(y_test, y_score)
hist_resids(y_test, y_score)  
resid_qq(y_test, y_score) 
resid_plot(y_test, y_score) 


### 
from sklearn.neighbors import KNeighborsRegressor
nr.seed(444)
KN_reg = KNeighborsRegressor(n_neighbors=7, leaf_size=35)
KN_reg.fit(Features_reduced, Labels)
y_score = KN_reg.predict(X_test_reduced)

print_metrics(y_test, y_score)
hist_resids(y_test, y_score)  
resid_qq(y_test, y_score) 
resid_plot(y_test, y_score) 

y_test
y_score



###============================================================ contrast
###============================================================ contrast

# Split train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Features, Labels_contrast, test_size=0.3, random_state=1122)

## Normalize the features
from sklearn.preprocessing import MinMaxScaler
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



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

print_metrics(y_test, y_score_l1)
hist_resids(y_test, y_score_l1)  
resid_qq(y_test, y_score_l1) 
resid_plot(y_test, y_score_l1) 


##### Try different model: linear_regression model
lin_mod = linear_model.LinearRegression()
lin_mod.fit(Features_reduced, Labels)
y_score = lin_mod.predict(X_test_reduced)

print_metrics(y_test, y_score)
hist_resids(y_test, y_score)  
resid_qq(y_test, y_score) 
resid_plot(y_test, y_score) 

#### Try different model: NN
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
#from statsmodels.api import datasets
from sklearn import datasets ## Get dataset from sklearn
import sklearn.model_selection as ms
import sklearn.metrics as sklm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy.random as nr

nr.seed(1115)
nn_mod = MLPRegressor(hidden_layer_sizes = (70,))
nn_mod.fit(Features_reduced, Labels)
y_score = nn_mod.predict(X_test_reduced)

print_metrics(y_test, y_score)
hist_resids(y_test, y_score)  
resid_qq(y_test, y_score) 
resid_plot(y_test, y_score) 



#### Try different model: 
from sklearn.ensemble import AdaBoostRegressor
nr.seed(444)
rf_regression = AdaBoostRegressor(learning_rate=0.5, random_state=1122)
rf_regression.fit(Features_reduced, Labels)
y_score = rf_regression.predict(X_test_reduced)

print_metrics(y_test, y_score)
hist_resids(y_test, y_score)  
resid_qq(y_test, y_score) 
resid_plot(y_test, y_score) 


### 
from sklearn.neighbors import KNeighborsRegressor
nr.seed(444)
KN_reg = KNeighborsRegressor(n_neighbors=7, leaf_size=35)
KN_reg.fit(Features_reduced, Labels)
y_score = KN_reg.predict(X_test_reduced)

print_metrics(y_test, y_score)
hist_resids(y_test, y_score)  
resid_qq(y_test, y_score) 
resid_plot(y_test, y_score) 

y_test
y_score






#===========================================================Highlights
# Split train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Features, Labels_highlights, test_size=0.3, random_state=1122)

## Normalize the features
from sklearn.preprocessing import MinMaxScaler
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



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

print_metrics(y_test, y_score_l1)
hist_resids(y_test, y_score_l1)  
resid_qq(y_test, y_score_l1) 
resid_plot(y_test, y_score_l1) 


##### Try different model: linear_regression model
lin_mod = linear_model.LinearRegression()
lin_mod.fit(Features_reduced, Labels)
y_score = lin_mod.predict(X_test_reduced)

print_metrics(y_test, y_score)
hist_resids(y_test, y_score)  
resid_qq(y_test, y_score) 
resid_plot(y_test, y_score) 

#### Try different model: NN
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
#from statsmodels.api import datasets
from sklearn import datasets ## Get dataset from sklearn
import sklearn.model_selection as ms
import sklearn.metrics as sklm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy.random as nr

nr.seed(1115)
nn_mod = MLPRegressor(hidden_layer_sizes = (70,))
nn_mod.fit(Features_reduced, Labels)
y_score = nn_mod.predict(X_test_reduced)

print_metrics(y_test, y_score)
hist_resids(y_test, y_score)  
resid_qq(y_test, y_score) 
resid_plot(y_test, y_score) 



#### Try different model: 
from sklearn.ensemble import AdaBoostRegressor
nr.seed(444)
rf_regression = AdaBoostRegressor(learning_rate=0.5, random_state=1122)
rf_regression.fit(Features_reduced, Labels)
y_score = rf_regression.predict(X_test_reduced)

print_metrics(y_test, y_score)
hist_resids(y_test, y_score)  
resid_qq(y_test, y_score) 
resid_plot(y_test, y_score) 


### 
from sklearn.neighbors import KNeighborsRegressor
nr.seed(444)
KN_reg = KNeighborsRegressor(n_neighbors=7, leaf_size=35)
KN_reg.fit(Features_reduced, Labels)
y_score = KN_reg.predict(X_test_reduced)

print_metrics(y_test, y_score)
hist_resids(y_test, y_score)  
resid_qq(y_test, y_score) 
resid_plot(y_test, y_score) 



#==============================================================Shadows

# Split train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Features, Labels_shadows, test_size=0.3, random_state=1122)

## Normalize the features
from sklearn.preprocessing import MinMaxScaler
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



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

print_metrics(y_test, y_score_l1)
hist_resids(y_test, y_score_l1)  
resid_qq(y_test, y_score_l1) 
resid_plot(y_test, y_score_l1) 


##### Try different model: linear_regression model
lin_mod = linear_model.LinearRegression()
lin_mod.fit(Features_reduced, Labels)
y_score = lin_mod.predict(X_test_reduced)

print_metrics(y_test, y_score)
hist_resids(y_test, y_score)  
resid_qq(y_test, y_score) 
resid_plot(y_test, y_score) 

#### Try different model: NN
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
#from statsmodels.api import datasets
from sklearn import datasets ## Get dataset from sklearn
import sklearn.model_selection as ms
import sklearn.metrics as sklm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy.random as nr

nr.seed(1115)
nn_mod = MLPRegressor(hidden_layer_sizes = (70,))
nn_mod.fit(Features_reduced, Labels)
y_score = nn_mod.predict(X_test_reduced)

print_metrics(y_test, y_score)
hist_resids(y_test, y_score)  
resid_qq(y_test, y_score) 
resid_plot(y_test, y_score) 



#### Try different model: 
from sklearn.ensemble import AdaBoostRegressor
nr.seed(444)
rf_regression = AdaBoostRegressor(learning_rate=0.5, random_state=1122)
rf_regression.fit(Features_reduced, Labels)
y_score = rf_regression.predict(X_test_reduced)

print_metrics(y_test, y_score)
hist_resids(y_test, y_score)  
resid_qq(y_test, y_score) 
resid_plot(y_test, y_score) 


### 
from sklearn.neighbors import KNeighborsRegressor
nr.seed(444)
KN_reg = KNeighborsRegressor(n_neighbors=7, leaf_size=35)
KN_reg.fit(Features_reduced, Labels)
y_score = KN_reg.predict(X_test_reduced)

print_metrics(y_test, y_score)
hist_resids(y_test, y_score)  
resid_qq(y_test, y_score) 
resid_plot(y_test, y_score) 















## Apply cross validation 
nr.seed(123)
inside = ms.KFold(n_splits=10, shuffle = True)
nr.seed(321)
outside = ms.KFold(n_splits=10, shuffle = True)

nr.seed(3456)
## Define the dictionary for the grid search and the model object to search on
param_grid = {"C": [0.1, 1, 10, 100, 1000]}
## Define the logistic regression model
logistic_mod = linear_model.LogisticRegression(class_weight = {0:0.45, 1:0.55}) 

## Perform the grid search over the parameters
clf = ms.GridSearchCV(estimator = logistic_mod, param_grid = param_grid, 
                      cv = inside, # Use the inside folds
                      scoring = 'roc_auc',
                      return_train_score = True)

## Fit the cross validated grid search over the data 
clf.fit(Features_reduced, Labels)

## And print the best parameter value
clf.best_estimator_.C
