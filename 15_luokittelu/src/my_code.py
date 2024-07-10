import sys
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


X=np.load('teach_data.npy')
Y=np.load('teach_class.npy')

################################################
#Your code below this line
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=200)

model = SVC()
model.fit(X_train, Y_train)

#Your code above this line
################################################

print('Compute real predictions')
real_X=np.load('data_in.npy')

print('real_X -', np.shape(real_X))
pred = model.predict(real_X)
print('pred -', np.shape(pred))
np.save('data_classified.npy', pred)
