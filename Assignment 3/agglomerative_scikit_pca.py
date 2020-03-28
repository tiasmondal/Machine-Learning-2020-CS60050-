import numpy as np
import sys
import csv
import pandas as pd
filename = "data_modified_new_numpy_from_net_pca.csv"
pd.options.mode.chained_assignment = None
with pd.option_context('display.precision', 10):
	data=pd.read_csv(filename,sep=r'\s*,\s*',header=0, encoding='ascii', engine='python')
data=np.array(data)
data=data[:,2:]
x=np.zeros((1,np.shape(data)[1]));
from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering(n_clusters=8,affinity='cosine',linkage='single').fit(data)
#print(clustering.labels_)
print(np.sort([i for i, x in enumerate(clustering.labels_) if x == 0]))
print([i for i, x in enumerate(clustering.labels_) if x == 1])
print([i for i, x in enumerate(clustering.labels_) if x == 2])
print([i for i, x in enumerate(clustering.labels_) if x == 3])
print([i for i, x in enumerate(clustering.labels_) if x == 4])
print([i for i, x in enumerate(clustering.labels_) if x == 5])
print([i for i, x in enumerate(clustering.labels_) if x == 6])
print([i for i, x in enumerate(clustering.labels_) if x == 7])