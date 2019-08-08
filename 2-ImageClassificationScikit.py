## Classification with SciKit pacakges Steps
#-> classify images to different folders
#? what if the images are not the same size???

#-> step 1: Prepare data for modeling
#-> step 2: build model
#-> step 3: fit model
#-> step 4: evaluate model
def prep_data (folder):
    # iterate through folders, assembling feature, label, and classname data objects
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    class_id = 0
    features = []
    labels = np.array([])
    classnames = []

    # dirs is file
    for root, dirs, filenames in os.walk(folder):
        for d in sorted(dirs):
            print("Reading data from ", d)
            # use the folder name as the class name for this label
            classnames.append(d)
            files = os.listdir(os.path.join(root,d))
            # f is the file names of each image in the dirs folder
            for f in files:
                # Load the image file
                # imageFile is the file path of each image in the dirs folder
                imgFile = os.path.join(root,d, f)
                img = plt.imread(imgFile) # use plt methods to read images

                ### to open NEF  files with rawpy
                #rp_image = rawpy.imread(imgFile)
                #img = rp_image.postprocess()
                # img is rgb file saved as np.array

                # The image array is a multidimensional numpy array
                # - flatten it to a single array of pixel values for scikit-learn
                # - and add it to the list of features

                features.append(img.ravel())
                # Add it to the numpy array of labels
                labels = np.append(labels, class_id )
            # after all images in the dirt folder is read, head to the next folder
            class_id  += 1
    # When all the images are read from the data file, convert feature list to np.array
    features = np.array(features)

    return features, labels, classnames
# The images are in a folder named 'shapes/training'
training_folder_name = 'C:/Users/EyesHigh/Desktop/ImageProcessing/data2/shapes/training'
features, labels, classnames = prep_data(training_folder_name)
print(len(features), 'features')
print(len(labels), 'labels')
print(len(classnames), 'classes:', classnames)
print(features.shape) # Feature Shape: (1200, 49152)
print(labels.shape) # Labels Shape: (1200,)

## step1-1 Split Data
# split into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.30)

print('Training records:',Y_train.size)
print('Test records:',Y_test.size)

## step2: Build the classification model
# step2.1 Normalize the pixel values
# step2-2 build model using Decision Tree algorithm
# Train the model
# -->> build pipeline with scaler and classifer 
# -->> pipeline.fit: fit pipeline with train data and save the fited model as clf
# -->> clf.predict: evaluate clf with test data
# -->> print merit metrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

# Convert the training features to floats so they can be scaled
X_train_float = X_train.astype('float64') # or can use: X_train_float = X_train*1. 
# Our pipeline performs two tasks:
#   1. Normalize the image arrays
#   2. Train a classification model
img_pipeline = Pipeline([('norm', MinMaxScaler()),
                         ('classify', DecisionTreeClassifier()),
                        ])

# step 3: Use the pipeline to fit a model to the training data
print("Training model...")
clf = img_pipeline.fit(X_train_float, Y_train)
print('classifier trained!')

## Step 4: Evaluate the model
# Evaluate classifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Convert the test features for scaling
X_test_float = X_test.astype('float64')

print('Classifier Metrics:')
predictions = clf.predict(X_test)
print(metrics.classification_report(Y_test, predictions, target_names=classnames))
print('Accuracy: {:.2%}'.format(metrics.accuracy_score(Y_test, predictions)))

print("\n Confusion Matrix:")
cm = confusion_matrix(Y_test, np.round(predictions, 0))
# Plot confusion matrix as heatmap
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(classnames))
plt.xticks(tick_marks, classnames, rotation=85)
plt.yticks(tick_marks, classnames)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

print(type(predictions)) # prediction is np.array of classes


## step 5: save the model and use it later to predict classes from new images
# Save the trained model
import sys
import os
import pickle

print ("Exporting the model")
file_stream = open('shape_classifier.pkl', 'wb')
pickle.dump(clf, file_stream)
file_stream.close()

## Use the model with new data
#-->> resize images to the same size
#-->> apply the model to predict the classes

# Helper function to resize image
def resize_image(src_img, size=(128,128), bg_color="white"): 
    from PIL import Image

    # rescale the image so the longest edge is the right size
    src_img.thumbnail(size, Image.ANTIALIAS)
    
    # Create a new image of the right shape
    new_image = Image.new("RGB", size, bg_color)
    
    # Paste the rescaled image onto the new background
    new_image.paste(src_img, (int((size[0] - src_img.size[0]) / 2), int((size[1] - src_img.size[1]) / 2)))
  
    # return the resized image
    return new_image

# Function to predict the class of an image
def predict_image(classifier, image_array):
    import numpy as np
    
    # These are the classes our model can predict
    classnames = ['circle', 'square', 'triangle']
    
    # Predict the class of each input image
    predictions = classifier.predict(image_array)
    
    predicted_classes = []
    for prediction in predictions:
        # And append the corresponding class name to the results
        predicted_classes.append(classnames[int(prediction)])
    # Return the predictions
    return predicted_classes # prediction_classes are names rather than number

import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
%matplotlib inline

# Load the model
print("Importing the model")
file_stream = open('shape_classifier.pkl', 'rb')
clf = pickle.load(file_stream)
file_stream.close()

#get the list of test image files
test_folder = 'C:/Users/EyesHigh/Desktop/ImageProcessing/data2/shapes/test'
test_image_files = os.listdir(test_folder)

# Empty array on which to store the images
image_arrays = []

size = (128,128)
background_color = "white"

fig = plt.figure(figsize=(12, 8))

# Get the images and show the predicted classes
for file_idx in range(len(test_image_files)):
    # go throught the image folder and store the images in list
    # read image as PIL
    img = Image.open(os.path.join(test_folder, test_image_files[file_idx]))

    # resize the image so it matches the training set - it  must be the same size as the images on which the model was trained
    resized_img = np.array(resize_image(img, size, background_color))
    # resize_image function return a PIL image, need to convert to array
    img_shape = np.array(resized_img).shape
    print(img_shape)

    # Add the image to the array of images
    image_arrays.append(resized_img.ravel())

# Get predictions from the array of image arrays
# Note that the model expects an array of 1 or more images - just like the batches on which it was trained
predictions = predict_image(clf, np.array(image_arrays))

# plot easch image with its corresponding prediction
for idx in range(len(predictions)):
    a=fig.add_subplot(1,len(predictions),idx+1)    
    img = image_arrays[idx].reshape(img_shape) # iamges saved in image_arrays is 1D, size of (49152,), should be covert to 3D to plot
    imgplot = plt.imshow(img)
    a.set_title(predictions[idx])