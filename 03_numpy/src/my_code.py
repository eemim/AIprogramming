import numpy as np

a = np.load('a.npy')
b = np.load('b.npy')

np. savez('ab.npz', a=a, b=b)
