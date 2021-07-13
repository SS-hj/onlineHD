import numpy as np
import pandas as pd

datapath='C:/onlinehd/proj_data/isolet1+2+3+4.data' #training data
original_data=pd.read_csv(datapath,sep=',',
                          names=['f{}'.format(x) for x in range(1,619)])
print(original_data.shape) #(6238, 618)

original_data.rename(columns={'f618':'y'},inplace=True)
print(original_data.head())

print(original_data['y'].nunique()) #26

datapath='C:/onlinehd/proj_data/isolet5.data' #test data
test_data=pd.read_csv(datapath,sep=',',
                          names=['f{}'.format(x) for x in range(1,619)])
print(test_data.shape) #(1559, 618)
test_data.rename(columns={'f618':'y'},inplace=True)
test_data.head()
print(test_data['y'].nunique()) #26

original_data.to_csv("C:/onlinehd/proj_data/train_set.csv",index=None)
test_data.to_csv("C:/onlinehd/proj_data/test_set.csv",index=None)