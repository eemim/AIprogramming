import pandas 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import svm

filename1='grading.csv'
train_fraction = 0.8
Y_column='Passed'

#Load data
data=pandas.read_csv(filename1)
print("Read data shape = "+str(data.shape))
print(data)


#################################################
#Your code here
#
#Create a classifier that classifies students
data.drop(['Name'], axis=1, inplace=True)
data['Passed'] = data['Passed'].astype(int)

def splitXY(d):
    return d.drop(Y_column, axis=1), d[Y_column]

traindata, testdata = train_test_split(data, test_size=1-train_fraction)
trainX, trainY=splitXY(traindata)
testX, testY=splitXY(testdata)

trainX=trainX.to_numpy()
testX=testX.to_numpy()

classifier = svm.SVC()
classifier.fit(trainX, trainY)
#
#################################################

#Load real data
filename2='assignments.csv'
data=pandas.read_csv(filename2)
names=data['Name']

#Remove name column
for col in ["Name"]:
    print("Remove "+col)
    data.drop(col, axis=1, inplace=True)
print()

print("Read data shape = "+str(data.shape))
print(data)
predY=classifier.predict(data)

#Create dataframe from numpy data
df = pandas.DataFrame({'Name': names, 'Passed': predY})
print(df)
df.to_csv('prediction.csv', index=False)

