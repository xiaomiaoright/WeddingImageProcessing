# WeddingImageProcessing

1. 1-ImageCompareRMSE.py:
Compare two rgb images(RGB) and returns the RMSE value

2. 1-BasicImageManipulations.py
Created functions to:
1)NEF2RGB: Convert NEF to RGB (np.array)
2)NEF2PIL: Convert NEF to PIL image
3)resize_image: resize the PIL image to target size, keep the scaling and add a background. (other resize methods can be considered, such as do not keep the scale of the original image. Need to decide which resize method to use before train the CNN model)
4)save_image_PIL: use PIL.save to save image
5)save_image_sk: use scikit-image imsave method
6)save_image_cv: use opencv imwrite method
7)save_image_plt: use matplotlib imsave method
8)imageNormalization: to normalize the RGB np.arrays to 0 to 1

3. 1-Contrast-Temperature.py
create functions to manipulate the image contrast and temperature. Need to figure out how the Exposure, Highlights, Shadows programmings

4. 2-ImageClassificationScikit.py
Create a CNN model with Scikit-image for image classification. The data preparation can be borrowed to our image processing project.

5. 2-ImageClassificationKeras.py
Create a CNN model with SKeras for image classification. 

6. 2-TransferLearning.py
Demoed how to load previous trained models of image feature extraction, add a fully connected layer for classification, and train the CNN model.

7. 3-ReadXMP2Labels.py
Create functions to read through the XMP files and extract "Exposure", "Constrast", "Highlight", "Shadows", "Temperature" parameters of each XMP file. Then save these parameters in a list. 
