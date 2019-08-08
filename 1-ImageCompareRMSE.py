# calculate the RMSE of two arrays of the same size
import numpy as np

def rmse(predictions, targets): 
    return np.sqrt(np.mean((predictions-targets)**2))

# test the function
arr1 = np.array([[1,2],[1,2],[1,2]]) 
arr2 = np.array([2,3,5])

print(rmse(arr1, arr2))