import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense
import os
import sys

sys.setdefaultencoding('utf8')

Y_column=8

def splitXY(d):
    return d.drop(Y_column, axis=1), d[Y_column] 


data=pd.read_csv('traindata.csv', encoding='utf-8', header=None)
data.replace({'I':0, 'F':-1, 'M':1}, inplace=True)

#Your code here...

X, Y = splitXY(data)
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=200)

model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(8,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, verbose=1)

'''
model_save_path = os.path.abspath('kotilo.h5')
print("Model polku:", model_save_path)

model.save(model_save_path)
print("Jeejee.")
'''
model.save('kotilo.h5')
    
predY = np.round(model.predict(testX))
n_test = testY.shape[0]

predY=np.round(model.predict(testX)).reshape((n_test, ))


accepted_n=(np.abs(predY-testY)<=3).sum()
print('Correct predictions:', accepted_n, '/', n_test);

