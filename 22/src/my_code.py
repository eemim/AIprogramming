#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 15:31:48 2022

@author: kojape
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

#Split data frame to x and y - don't modify
def splitXY(d):
    Y_column='fetal_health'
    return d.drop(Y_column, axis=1).to_numpy(), np.int32(d[Y_column]) 


def load_data(filename):
    global drop_columns
    data=pd.read_csv(filename)
    for col in drop_columns:
        data.drop(col, axis=1, inplace=True)

    return data    

def init_model(datafile):
    
    global model
    global drop_columns
    global mean, std
    

    #Modify this to drop unused columns from data
    drop_columns=['baseline value', 'histogram_width', 'histogram_min', 'histogram_max', 
                  'histogram_number_of_peaks', 'histogram_number_of_zeroes', 'histogram_mode',
                  'histogram_mean', 'histogram_median', 'histogram_variance', 'histogram_tendency']

    data=load_data(datafile) #load dataframe
    data.dropna(inplace=True)
    
    X, Y = splitXY(data)
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=200)
    #your code here
    # - Preprocess data
    # - Split data into two pieces; train set and test set
    # - Create and train model
    # - y_test shall contain integer values in range 1..3
    # - Use global variables to store scaling information.
    #   You may need it in compute_value function.
    mean = trainX.mean(axis=0)
    std = trainX.std(axis=0)

   # Scale the training data
    trainX = (trainX - mean) / std
    
    x_test = testX
    y_test = testY
    model = SVC()
    model.fit(trainX, trainY)

    return x_test, y_test #return test set for testing purposes


#Compute predicted value of x
def compute_value(x):
    global model
    global mean, std
    
    x_scaled = (x - mean) / std
    #your code here
    # - x is unscaled x value, scale it
    # - utilize model to predict y_pred vector values, y_pred is vector which 
    #   elements are integers in 1..3
    y_pred=model.predict(x_scaled)
    
    
    return y_pred

#Loads validation data set and computes predictions based on x-values
#Don't utilize y-values except for testing purposes!
def validate(filename):
    data_val=load_data(filename)
    x_val, y_val=splitXY(data_val)
    y_pred=compute_value(x_val)
    return y_pred, y_val


#Test program (don't modify)
if __name__ == "__main__":
    x_test, y_test=init_model('fetal_health.csv')

    testN, _=np.shape(x_test)
    assert(np.shape(y_test)==(testN,))

    y_pred=compute_value(x_test)
    errors=(y_pred!=y_test).sum()
    print('Errors (test set):', errors, '/', testN)

    y_pred, y_val=validate('validation.csv')
    valN=np.shape(y_val)[0]
    errors=(y_pred!=y_val).sum()
    print('Errors (validation set):', errors, '/', valN)


    
    