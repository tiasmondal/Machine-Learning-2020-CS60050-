import numpy as np
import csv
import pandas as pd
from tqdm import tqdm
import math
filename = "data_modified_pca.csv"
data1=pd.read_csv(filename,sep=r'\s*,\s*',header=0, encoding='ascii', engine='python')

headers=data1.columns.values;
headers=headers[1:]
print(headers)

data=np.array(data1);
data=data[:,2:];
print(data)
# exit()
distance=np.zeros((len(data1["Religious_texts"]),len(data1['Religious_texts'])))
def norm(A):
	return((np.sum(A**2))**0.5)
# 	normsum=0;
# 	for i in range(0,len(A)):
# 		normsum=normsum+A[i]**2;
# 	return(normsum**0.5)
# k=1
# sum1=np.sum(data[k]**2)
# #sum1=sum1**0.5
# print(sum1)
# for i in range(0,len(data1['Religious_texts'])):
# 	print(norm(data[i]))
# 	if(str(norm(data[i]))=="nan"):
# 		print(i)
# exit()

for i in tqdm(range(len(data1['Religious_texts']))):
	for j in tqdm(range(len(data1['Religious_texts']))):
		x=norm(data[i]);
		y=norm(data[j]);
		if(str(x)!="nan" or str(y)!="nan"):
			if(round(((data[i].dot(data[j]))/(x*y)),4)>1):
				print("Not possible")
				print(data[i].dot(data[j]))
				print(x*y)
			dist=round(((data[i].dot(data[j]))/(x*y)),4);
			distance[i][j]=np.exp(-dist);
			#distance[i][j]=math.acos(dist);
		else:
			print("red alert "+str(i)+" "+str(j))
			distance[i][j]=np.exp(-1);
			#distance[i][j]=math.acos(1);

print(distance)

distance.dump("distance_new_inverse_cosine_pca.dat")

################################## Agglomerative clustering part ####################################################
# s=len(data1['Religious_texts'])
# clusters=[]

# for i in range(0,len(data1['Religious_texts'])):
# 	clusters.append([i]);

# while(len(clusters)!=8):
# 	ij_min = np.unravel_index(distance.argmin(), distance.shape);
# 	x=[];
	
# 	for i in range(0,len(clusters)):
# 		if(ij_min[0] in clusters[i]):
# 			x.append(i);
# 		if(ij_min[1] in cluster[i]):
# 			x.append(i);
# 	if(x[0]==x[1]):
# 		print("Something wrong");
# 		exit();
# 	for i in range(len(clusters[x[0]])):
# 		for j in range(len(clusters[x[1]])):
# 			distance[clusters[x[0]]][clusters[x[1]]]=10e7;
# 			distance[clusters[x[1]]][clusters[x[0]]]=10e7;
	
# 	clusters[x[0]]=clusters[x[0]]+clusters[x[1]];
# 	del clusters[x[1]];
#print(data[0])

# for i in range(0,np.shape(data)[1]):
# 	x=