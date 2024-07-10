import pandas as pd
from scipy import stats
import numpy as np
from sklearn import preprocessing


inputfile='time_series.csv'
trainfile='train.csv'
testfile='test.csv'

data=pd.read_csv(inputfile)

#Your code here
filtered_df = data[(data['region'] == 'DE') & (data['variable'] == 'wind') & (data['attribute'] == 'generation_actual')]
data = filtered_df.copy() 
#print("data shape = "+str(data.shape))
#print("data shape = "+str(filtered_df.shape))
#print(filtered_df)
#data.drop(filtered_df.index, inplace=True)
#print("data shape = "+str(data.shape))

data.dropna(inplace=True)
data.drop_duplicates(inplace=True)

z_threshold=4.0
z = np.abs(stats.zscore(data['data']))
filtered_entries = (z < z_threshold)

data = data[filtered_entries]

data.drop(['region', 'variable', 'attribute'], axis=1, inplace=True)
      
X_train_minmax = data['data'].values.reshape(-1, 1)
min_max_scaler = preprocessing.MinMaxScaler()
data['data'] = min_max_scaler.fit_transform(X_train_minmax)

train_fraction = 0.7

traindata=data.sample(frac=train_fraction,random_state=200)

testdata=data.drop(traindata.index)

#Save train data
#print("Save train data")
traindata.to_csv(trainfile)

#Save test data
#print("Save test data")
testdata.to_csv(testfile)
'''
print(testdata.columns)
print(32*'*')
print(traindata.columns)
print("Number of rows in train data:", len(traindata))
print("Number of rows in test data:", len(testdata))
'''