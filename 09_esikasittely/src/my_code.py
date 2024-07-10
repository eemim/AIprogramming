import sys
import time
import pandas as pd

inputfile='weather_data.csv'
outputfile='preprocessed.csv'

data=pd.read_csv(inputfile)

#Your code here
#print("data shape = "+str(data.shape))
data.drop(columns=['utc_timestamp'], inplace=True)
#print("data shape = "+str(data.shape))

data.dropna(inplace=True)
#print("data shape = "+str(data.shape))
data.to_csv(outputfile)
