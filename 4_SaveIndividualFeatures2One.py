# To create the a full dataframe of all images in wedding dataset.
# Append 4 categorical together. 
import os
import pandas as pd

ED_IP_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/ED_Label_IndoorPerson.csv"
ED_IT_path ="/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/ED_Label_IndoorThings.csv"
ED_OP_path ="/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/ED_Label_OutdoorPerson.csv"
ED_OT_path ="/Users/user7/Desktop/WeddingImageProcessing/WeddingDatasetBins32/ED_Label_OutdoorThings.csv"

ED_Labels_DF = pd.read_csv(ED_IP_path, index_col = 0) 
ED_Labels_DF.shape

EDLabelDf_path = [ED_IT_path, ED_OP_path, ED_OT_path]

for labelDf in EDLabelDf_path:
    dataset = pd.read_csv(labelDf, index_col = 0) 
    ED_Labels_DF = ED_Labels_DF.append(dataset)

ED_Labels_DF.shape

# read the ED Features 
ED_Features_bins16_path = "/Users/user7/Desktop/WeddingImageProcessing/WeddingDataPrepBins16/ED_Feature.csv"
ED_Features_bins16 = pd.read_csv(ED_Features_bins16_path, index_col = 0) 

ED_Features_bins16.shape


## join the features and labels

ED_Features_Labels_Bins16 = ED_Features_bins16.join(ED_Labels_DF, how = "inner")
ED_Features_Labels_Bins16.shape
ED_Features_Labels_Bins16.columns