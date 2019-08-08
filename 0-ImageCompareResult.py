# Image from cloud: EDWedding file: 800_1743.NEF
# To calculate the image chage in terms of np.array and RMSE after editting actions

# Images editted in LightRoom, one chagne one parater at a time
# test_C1: increase constrast by 1, keep other parameter unchanged
# test_E005: increase exposure by 0.05, keep other parameter unchanged
# test_H1: increase highlights by 1, keep other parameter unchanged
# test_S1: increase shadows by 1, keep other parameter unchanged
# test_T3751: increase temperature by 1, keep other parameter unchanged
import numpy as np
import skimage as sk
from skimage import io as sk_io


def compare_image(img1_path, img2_path):
    img1 = sk_io.imread(img1_path)
    img2 = sk_io.imread(img2_path)
    img1_normalized = img1 * 1. / 255.
    img2_mormalized = img2 * 1. / 255.
    rmse = np.sqrt(np.mean((img1_normalized-img2_mormalized)**2))

    return rmse

# test  
test_path = "//Users/user7/Desktop/Data/EDWedding/Test/test/test/test.jpg" # original image
test_T3751_path = "/Users/user7/Desktop/Data/EDWedding/Test/test/test/test_T3751.jpg"
test_C1_path = "/Users/user7/Desktop/Data/EDWedding/Test/test/test/test_C1.jpg"
test_E005_path = "/Users/user7/Desktop/Data/EDWedding/Test/test/test/test_E005.jpg"
test_H1_path = "/Users/user7/Desktop/Data/EDWedding/Test/test/test/test_H1.jpg"
test_S1_path = "/Users/user7/Desktop/Data/EDWedding/Test/test/test/test_S1.jpg"




rmse_C1 = compare_image(test_C1_path,test_path) 
print("Contrast change 1: ", rmse_C1)  

rmse_T3751 = compare_image(test_T3751_path, test_path)
print("Temperature change 1: ", rmse_T3751)

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