# Image from cloud: EDWedding file: CAR_4186.NEF
# To calculate the image chage in terms of np.array and RMSE after editting actions

# Images editted in LightRoom, one chagne one parater at a time
# test_C1: increase constrast by 1, keep other parameter unchanged
# test_E005: increase exposure by 0.05, keep other parameter unchanged
# test_H1: increase highlights by 1, keep other parameter unchanged
# test_S1: increase shadows by 1, keep other parameter unchanged
# test_T50: increase temperature by 1, keep other parameter unchanged
import numpy as np
import skimage as sk
from skimage import io as sk_io
import matplotlib.pyplot as plt
from matplotlib import image as mp_image

def compare_image(img1_path, img2_path):
    img1 = sk_io.imread(img1_path)
    img2 = sk_io.imread(img2_path)
    img1_normalized = img1 * 1. / 255.
    img2_mormalized = img2 * 1. / 255.
    rmse = np.sqrt(np.mean((img1_normalized-img2_mormalized)**2))

    return rmse

def image_compare_plt(img1_path, img2_path, editting):
    
    img1 = sk_io.imread(img1_path)
    img2 = sk_io.imread(img2_path)

    fig = plt.figure(figsize=(12, 12))

    # Subplot for original image
    a=fig.add_subplot(3,2,1)
    imgplot = plt.imshow(img1)
    a.set_title('Original')

    # Subplot for editted image
    a=fig.add_subplot(3,2,2)
    imgplot = plt.imshow(img2)
    a.set_title(editting)

    # Subplots for histograms
    a=fig.add_subplot(3,2,3)
    imgplot = plt.hist(img1.ravel())

    a=fig.add_subplot(3,2,4)
    imgplot = plt.hist(img2.ravel())

    # Subplots for CDFs
    a=fig.add_subplot(3,2,5)
    imgplot = plt.hist(img1.ravel(), bins=255, cumulative=True)

    a=fig.add_subplot(3,2,6)
    imgplot = plt.hist(img2.ravel(), bins=255, cumulative=True)

    plt.show()

# test  
test_path = "/Users/user7/Desktop/WeddingImageProcessing/DataExplore/CAR_4186.jpg" # original image
test_T50_path = "/Users/user7/Desktop/WeddingImageProcessing/DataExplore/CAR_4186-T50.jpg"
test_C1_path = "/Users/user7/Desktop/WeddingImageProcessing/DataExplore/CAR_4186-C1.jpg"
test_E005_path = "/Users/user7/Desktop/WeddingImageProcessing/DataExplore/CAR_4186-E005.jpg"
test_H1_path = "/Users/user7/Desktop/WeddingImageProcessing/DataExplore/CAR_4186-H1.jpg"
test_S1_path = "/Users/user7/Desktop/WeddingImageProcessing/DataExplore/CAR_4186-S1.jpg"

# Save path and edit operations to a list
path = [test_path, test_T50_path, test_C1_path, test_E005_path, test_H1_path, test_S1_path]
edit = ["original", "Contrast+1", "Exposure+0.05", "Highlights+1", "Shadows+1", "Temperature+50"]

# Print the RMSE change after Editting
for i in range(5):
    rmse = compare_image(path[0], path[i+1])
    print(edit[i+1], rmse)
    
# Plot the compare image and original images
for i in range(5):
    image_compare_plt(path[0], path[i+1], edit[i+1])


rmse_C1 = compare_image(test_path,test_C1_path) 
print("Contrast change 1: ", rmse_C1)  
image_compare_plt(test_path,test_C1_path, "Contrast+1")

img1 = sk_io.imread(test_path)
img2 = sk_io.imread(test_C1_path)

change = img1 - img2
print(change[0][:10])


rmse_T50 = compare_image(test_T50_path, test_path)
print("Temperature change 1: ", rmse_T50)

rmse_E005 = compare_image(test_E005_path,test_path)
print("Exposure change 0.05: ", rmse_E005)

rmse_H1 = compare_image(test_H1_path,test_path)
print("Highlights change 1: ", rmse_H1)

rmse_S1 = compare_image(test_S1_path, test_path)
print("Shadow change 1: ", rmse_S1)






"""
Result:
Contrast change 1:  0.0044245886069989
Temperature change 1:  0.003812718388979996
Exposure change 0.05:  0.00975518159382403
Highlights change 1:  0.0036392888264328823
Shadow change 1:  0.003856666570035857
"""