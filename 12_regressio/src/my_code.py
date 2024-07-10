import sys
import time
import pandas
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

inputfile='mittaus.csv'


data=pandas.read_csv(inputfile)


#Your code here

data_x = data['x'].values.reshape(-1, 1)
data_y = data['y'].values.reshape(-1, 1)


model_degree = 2


poly_reg = PolynomialFeatures(degree=model_degree)
X_poly = poly_reg.fit_transform(data_x)


linreg2 = linear_model.LinearRegression()
linreg2.fit(X_poly, data_y)

a = linreg2.coef_[0, 2]
b = linreg2.coef_[0, 1]
c = linreg2.intercept_[0]

discriminant = b**2 - 4*a*c
landing_spot_x = (-b - np.sqrt(discriminant)) / (2*a)

print(f'{landing_spot_x:.1f}')

