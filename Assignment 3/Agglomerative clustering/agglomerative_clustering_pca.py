import numpy as np
import csv
import pandas as pd


distance=np.random.random((10,10))
#distance=[[1,0.9,0.1,0.65,0.2],[0.9,1,0.7,0.6,0.5],[0.1,0.7,1,0.4,0.3],[0.65,0.6,0.4,1,0.8],[0.2,0.5,0.3,0.8,1]];
#distance=[[0,0.71,5.66,3.61,4.24,3.20],[0.71,0,4.95,2.92,3.54,2.5],[5.66,4.95,0,2.24,1.41,2.50],[3.61,2.92,2.24,0,1,0.5],[4.24,3.54,1.41,1,0,1.12],[3.2,2.5,2.5,0.5,1.12,0]];
distance = np.load("distance_new_inverse_cosine_pca.dat", allow_pickle=True)
#distance=[[0,9,3,6,11],[9,0,7,5,10],[3,7,0,9,2],[6,5,9,0,8],[11,10,2,8,0]];
distance=np.array(distance)
print(distance)
# s=len(np.shape(distance)[0])
clusters=[]


for i in range(0,np.shape(distance)[0]):
	clusters.append([i]);

distance[np.diag_indices_from(distance)]=10e7;
while(len(clusters)!=8):
	ij_min = np.unravel_index(distance.argmin(), distance.shape);
	x=[];
	for i in range(0,len(clusters)):
		if(ij_min[0] in clusters[i]):
			x.append(i);
		if(ij_min[1] in clusters[i]):
			x.append(i);
	if(x[0]==x[1]):
		print("Something wrong");
		print(distance)
		print(x[0])
		print(ij_min[0])
		print(ij_min[1])
		exit();
	for i in range(len(clusters[x[0]])):
		for j in range(len(clusters[x[1]])):
			distance[clusters[x[0]][i]][clusters[x[1]][j]]=10e7;  #Making joined point distances inf so that,
			distance[clusters[x[1]][j]][clusters[x[0]][i]]=10e7;  #they re not counted as min in next iteration
	
	clusters[x[0]]=clusters[x[0]]+clusters[x[1]];
	del clusters[x[1]];
	#print(len(clusters))

#print(clusters)

print(np.sort(clusters[0]))
print(clusters[1])
print(clusters[2])
print(clusters[3])
print(clusters[4])
print(clusters[5])
print(clusters[6])
print(clusters[7])

for i in range(len(clusters)):
	clusters[i]=np.sort(clusters[i]);
clusters.sort(key=lambda x:min(x))
np.savetxt("cluster_agglomerative_pca.txt",clusters,fmt="%s");
np.array(clusters).dump("clusters_agglomerative_pca.dat");