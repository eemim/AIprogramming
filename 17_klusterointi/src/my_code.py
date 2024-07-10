import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


data = pd.read_csv("painteddata.csv")

#Your code here

n_clusters_array=[i for i in range(2, 10)]
silhouette_avg=[]

for clusters in n_clusters_array: 
    kmeans=KMeans(n_clusters=clusters).fit(data)
    kmeans_classID=kmeans.labels_
 
    silhouette_avg.append(silhouette_score(data, kmeans_classID))
    
max_idx=silhouette_avg.index(max(silhouette_avg))
n_clusters = n_clusters_array[max_idx]
print(n_clusters)
