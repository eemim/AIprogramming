import numpy as np

tiedosto = np.load('m4in.npz')

b = tiedosto['b']

np.save('m4out.npy', b)