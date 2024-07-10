import numpy as np

a = np.load('m2in.npy')

a[2, 3] = -1.0

np.save('m2out.npy', a)