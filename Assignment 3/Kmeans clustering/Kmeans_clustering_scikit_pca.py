from sklearn.cluster import KMeans
import numpy as np
import numpy as np
import csv
import pandas as pd
from tqdm import tqdm
import math
import random

#X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])


filename = "data_modified_pca.csv"
pd.options.mode.chained_assignment = None
with pd.option_context('display.precision', 10):
	data=pd.read_csv(filename,sep=r'\s*,\s*',header=0, encoding='ascii', engine='python')
data=np.array(data)
data=data[:,2:]
# kmeans.predict([[0, 0], [12, 3]])
# kmeans.cluster_centers_
clustering = KMeans(n_clusters=8, random_state=0, init='k-means++',max_iter=300).fit(data);
print(np.sort([i for i, x in enumerate(clustering.labels_) if x == 0]))
print([i for i, x in enumerate(clustering.labels_) if x == 1])
print([i for i, x in enumerate(clustering.labels_) if x == 2])
print([i for i, x in enumerate(clustering.labels_) if x == 3])
print([i for i, x in enumerate(clustering.labels_) if x == 4])
print([i for i, x in enumerate(clustering.labels_) if x == 5])
print([i for i, x in enumerate(clustering.labels_) if x == 6])
print([i for i, x in enumerate(clustering.labels_) if x == 7])

#print(clustering.cluster_centers_)
clustering.cluster_centers_.dump("Initial_centroids_Kmeans++.dat");