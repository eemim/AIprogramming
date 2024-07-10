import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

inputfile='in.npy'
outputfile='out.npy'

data=np.load(inputfile)

#Your code here

pca = PCA()

pca.fit(data)

'''
plt.plot(pca.explained_variance_)
plt.show()

plt.plot(pca.explained_variance_)
plt.semilogy()
plt.show()
'''
explained_variance_ratio = pca.explained_variance_ratio_
required_variance_ratio = 0.1

for n_components in range(20, 23):
    pca = PCA(n_components=n_components)
    pca.fit(data)
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    if cumulative_variance_ratio[-1] >= required_variance_ratio:
        break

packed_data = pca.transform(data)


np.save(outputfile, packed_data)

#print(data.shape)
#print(packed_data.shape)


