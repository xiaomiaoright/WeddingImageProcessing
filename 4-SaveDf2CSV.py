# to save the dataframe (features/label dataframe) to csv fies
# csv files can be shared among group member for data exploration

# csv_file_name need to have .csv, eg: LC_IndoorPerson.csv
import os
import pandas as pd

def DataFrame2CSV(df, csv_folder_path, csv_file_name):
    
    csv_file_path = os.path.join(csv_folder_path, csv_file_name)
    df.to_csv(csv_file_path)

# test
Cars = {'Brand': ['Honda Civic','Toyota Corolla','Ford Focus','Audi A4'],
        'Price': [22000,25000,27000,35000]
        }

df = pd.DataFrame(Cars, columns= ['Brand', 'Price'])

print(df)

test_csv_path = "/Users/user7/Desktop/WeddingImageProcessing/data_test"
test_csv_name = "CarPrice.csv"
DataFrame2CSV(df, test_csv_path, test_csv_name)
