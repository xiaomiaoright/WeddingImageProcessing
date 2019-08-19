# define the plots 
import sklearn
import sklearn.metrics as sklm
import seaborn as sns
import math
import scipy.stats as ss
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


import pandas as pd
!pip install --upgrade keras
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import pandas as pd


import keras
from keras import backend as K

print('Keras version:',keras.__version__)

# Load the dataframe
train_path = "/home/ubuntu/Documents/DataPrep/CNN_v1/CNN_ED_KJ_JY_LM_jpg_DF.csv"
Train_DF = pd.read_csv(train_path, index_col = 0)

## Use 2000 samples for training and validation, use 433 samples for test.
import sklearn.model_selection as ms
from sklearn.model_selection import train_test_split

X_train, X_test, X_Validate, y_test = train_test_split(Train_DF, Labels_DF, test_size=0.3, random_state=1122)


### ===================>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
### ===================>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
### ===================>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Build CNN manually
datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)

## Prepare the data
train_iter=datagen.flow_from_dataframe(
dataframe=X_train,
directory=None,
x_col="JPG_File_Path",
y_col='Exposure',
has_ext=False,
subset="training",
class_mode="other",
target_size=(80,120))

valid_iter=datagen.flow_from_dataframe(
dataframe=X_train,
directory=None,
x_col="JPG_File_Path",
y_col='Exposure',
has_ext=False,
subset="validation",
class_mode="other",
target_size=(80,120))

# Define CNN
## The images has to be jpg, ## use 60 (3x3) filters 
model = Sequential()
model.add(Conv2D(40, (3, 3), padding='same',
                 input_shape=(80,120,3)))
model.add(Activation('relu'))
model.add(Conv2D(40, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='tanh'))
model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss="mse",metrics=["mse"])

model.summary()


#Train CNN
STEP_SIZE_TRAIN=train_iter.n//train_iter.batch_size
STEP_SIZE_VALID=valid_iter.n//valid_iter.batch_size
history4 = model.fit_generator(generator=train_iter,
                    use_multiprocessing=True,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_iter,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=50
)

# Plot result
%matplotlib inline
from matplotlib import pyplot as plt

epoch_nums = range(1,50+1)
training_loss = history4.history["loss"]
validation_loss = history4.history["val_loss"]
plt.plot(epoch_nums, training_loss)
plt.plot(epoch_nums, validation_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()


# plot MSE
epoch_nums = range(1,51)
training_MSE = history4.history["mean_squared_error"]
validation_MSE = history4.history['val_mean_squared_error']
plt.plot(epoch_nums, training_MSE)
plt.plot(epoch_nums, validation_MSE)
plt.xlabel("epoch")
plt.ylabel("MSE")


## Test the model
datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)

## Prepare the data
test_iter=datagen.flow_from_dataframe(
dataframe=X_test,
directory=None,
x_col="JPG_File_Path",
y_col='Exposure',
has_ext=False,
subset=None,
class_mode="other",
target_size=(80,120))

y_score =model.predict_generator(test_iter, 
                            steps = test_iter.n//test_iter.batch_size+1,
                            max_queue_size=50, 
                            workers=1, 
                            use_multiprocessing=True, 
                            verbose=0)
print_metrics(y_test, y_score)
hist_resids(y_test, y_score)  
resid_qq(y_test, y_score) 
resid_plot(y_test, y_score) 



### ===================>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
### ===================>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
### ===================>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Transfer Learn:  VGG
from keras import applications16
#Load the base model, not including its final connected layer, and set the input shape to match our images
base_model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=train_iter.image_shape)

## Freeze the trained layers 
from keras import Model
from keras.layers import Flatten, Dense
from keras import optimizers

# Freeze the already-trained layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Create layers for last regression of our images
x = base_model.output
x = Flatten()(x)
prediction_layer = Dense(1, activation='tanh')(x) 
model = Model(inputs=base_model.input, outputs=prediction_layer)

model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss="mse",metrics=["mse"])

print(model.summary())

# Fit the model
STEP_SIZE_TRAIN=train_iter.n//train_iter.batch_size
STEP_SIZE_VALID=valid_iter.n//valid_iter.batch_size
history5 = model.fit_generator(generator=train_iter,
                    use_multiprocessing=True,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_iter,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=15
)

%matplotlib inline
from matplotlib import pyplot as plt

epoch_nums = range(1,15+1)
training_loss = history5.history["loss"]
validation_loss = history5.history["val_loss"]
plt.plot(epoch_nums, training_loss)
plt.plot(epoch_nums, validation_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()

# plot MSE
epoch_nums = range(1,16)
training_MSE = history4.history["mean_squared_error"]
validation_MSE = history4.history['val_mean_squared_error']
plt.plot(epoch_nums, training_MSE)
plt.plot(epoch_nums, validation_MSE)
plt.xlabel("epoch")
plt.ylabel("MSE")


## Test the model
datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)

## Prepare the data
test_iter=datagen.flow_from_dataframe(
dataframe=X_test,
directory=None,
x_col="JPG_File_Path",
y_col='Exposure',
has_ext=False,
subset=None,
class_mode="other",
target_size=(80,120))

y_score =model.predict_generator(test_iter, 
                            steps = test_iter.n//test_iter.batch_size+1,
                            max_queue_size=50, 
                            workers=1, 
                            use_multiprocessing=True, 
                            verbose=0)
print_metrics(y_test, y_score)
hist_resids(y_test, y_score)  
resid_qq(y_test, y_score) 
resid_plot(y_test, y_score) 


### ===================>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
### ===================>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
### ===================>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Transfer Learn:  InceptionV3

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# Freeze the already-trained layers in the base model
for layer in base_model.layers:
    layer.trainable = False
    
# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(1, activation='tanh')(x)

# this is the model we will train

print(model.summary())

STEP_SIZE_TRAIN=train_iter.n//train_iter.batch_size
STEP_SIZE_VALID=valid_iter.n//valid_iter.batch_size
history5 = model.fit_generator(generator=train_iter,
                    use_multiprocessing=True,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_iter,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=50
)