# WeddingImageProcessing

1. ImageCOmpareRMSE.py:
Compare two rgb images(RGB) and returns the RMSE value

2. BasicImageManipulations.py
Created functions to:
1)NEF2RGB: Convert NEF to RGB (np.array)
2)NEF2PIL: Convert NEF to PIL image
3)resize_image: resize the PIL image to target size, keep the scaling and add a background
4)save_image_PIL: use PIL.save to save image
5)save_image_sk: use scikit-image imsave method
6)save_image_cv: use opencv imwrite method
7)save_image_plt: use matplotlib imsave method
8)imageNormalization: to normalize the RGB np.arrays to 0 to 1