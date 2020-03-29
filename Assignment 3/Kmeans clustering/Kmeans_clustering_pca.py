import numpy as np
import csv
import pandas as pd
from tqdm import tqdm
import math
import random


filename = "data_modified_pca.csv"
pd.options.mode.chained_assignment = None
with pd.option_context('display.precision', 10):
	data=pd.read_csv(filename,sep=r'\s*,\s*',header=0, encoding='ascii', engine='python')
data=np.array(data)
data=data[:,2:]
centroid=[];
centroid_new=[];
# max_col=[];
# min_col=[];
#data=np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15],[16,17,18]]);
# for i in range(np.shape(data)[1]):
# 	max_col.append(np.reshape(max(data[:,i:i+1]),()));
# 	min_col.append(np.reshape(min(data[:,i:i+1]),()));
# # print(np.shape(max_col))
# # print(np.shape(min_col));
# # exit()
# max_col=np.array(max_col)
# min_col=np.array(min_col)
# average_col=[];
# average_col=(max_col-min_col)/8

def norm(A):
	return((np.sum(A**2))**0.5);

# for i in range(0,8):
# 	centroid.append(min_col+i*average_col);
# 	centroid_new.append(min_col+i*average_col);
# centroid[0][0]=1;
# centroid_new[0][0]=1;
#centroid=np.random.random((8,8266))


# print(np.shape(centroid))
# print(np.shape(data[0]))
# exit()
#print(centroid)

# data_mod=np.zeros(8266,)
# for i in range(0,590):
# 	print(norm(data[i]));
# exit()

it=0
print("Randomly selecting 8 centroids, Please wait.............")
t_m=0;
while(1):
	random_sample=random.sample(range(589), 8)
	clusters=[[],[],[],[],[],[],[],[]]
	centroid=[];
	centroid_new=[];

	for i in range(8):
		centroid.append(data[random_sample[i]]);
		centroid_new.append(data[random_sample[i]]);
#centroid=np.ones((8,8266))
	
	for j in range(589):              #change
		distance=[];
		for i in range(0,8):
			distance.append(np.exp(-round(((centroid[i].dot(data[j]))/(norm(centroid[i])*norm(data[j]))),4))); #Cosine Dist
			#distance.append(round(((np.sum((np.square(centroid[i]-data[j]))))**0.5),4));    #Euclidian Dist
		clusters[np.argmin(distance)].append(j);


	count1=0;
	for i in range(len(clusters)):
		if(len(clusters[i])!=0):
			count1=count1+1;


	if(count1==8):
		break;
	print(count1)
	t_m=t_m+1;
	if(t_m>3): #if random initialization fails
		print("Choosing Centroids..........")
		centroid = np.load("Initial_centroids_Kmeans++.dat", allow_pickle=True)   #Initial Centroid choosing using Kmeans++ algo
		centroid_new=centroid


#print(clusters)





#centroid = np.load("Initial_centroids_Kmeans++.dat", allow_pickle=True)   #Initial Centroid choosing using Kmeans++ algo
# centroid_new=centroid
print("Successfully selected 8 centroids")
np.array(clusters).dump("clusters_k_means_pca.dat");
for i in range(len(clusters)):
	clusters[i]=np.sort(clusters[i])
clusters.sort(key=lambda x:min(x))
np.savetxt("clusters_k_means_pca.txt",clusters,fmt="%s");
while(it>=0):
	clusters=[[],[],[],[],[],[],[],[]]
	it=it+1;
	for j in range(589):       #change
		distance=[];
		for i in range(0,8):
			distance.append(np.exp(-round(((centroid[i].dot(data[j]))/(norm(centroid[i])*norm(data[j]))),4))); #Cosine Dist
			#distance.append(round(((np.sum((np.square(centroid[i]-data[j]))))**0.5),4));  #Euclidian Distance
		clusters[np.argmin(distance)].append(j); 
	
	count1=0;
	for i in range(len(clusters)):
		if(len(clusters[i])!=0):
			count1=count1+1;
	if(count1!=8):
		print("Reason 1")
		break;
	print(clusters)
	for i in range(0,8):
		sum1=np.zeros(len(data[0]),);
		for j in range(0,len(clusters[i])):
			# print(clusters[i][j]);
			# print(np.shape(data[clusters[i][j]]));
			# print(np.shape(sum1))
			sum1=sum1+data[clusters[i][j]]
		#print(len(clusters[i]))
		#print(sum1/len(clusters[i]))
		centroid_new[i]=sum1/len(clusters[i]);
		
		# print()

	count=0;
	# print(np.array(centroid)-np.array(centroid_new));
	# exit()
	print("centroid")
	print(centroid)
	print("centroid_new")
	print(centroid_new)
	count_equality=0;
	for i in range(0,len(centroid)):
		for j in range(0,len(centroid[i])):
			if(centroid[i][j]==centroid_new[i][j]):
				count_equality=count_equality+1;
	print("count_equality");
	print(count_equality)
	if(count_equality==800):
		print(clusters)
		np.array(clusters).dump("clusters_k_means_pca.dat")
		for i in range(len(clusters)):
			clusters[i]=np.sort(clusters[i])
		clusters.sort(key=lambda x:min(x))
		np.savetxt("clusters_k_means_pca.txt",clusters,fmt="%s");
		print("Number of iterations "+str(it));
		print("Reason 2")
		exit()

	# for i in range(0,8):
	# 	x=round(((centroid[i].dot(centroid_new[i]))/(norm(centroid[i])*norm(centroid_new[i]))),4)
	# 	print(abs(math.acos(x)))
	# 	if(abs(math.acos(x))<=0.01):
	# 		count=count+1;
	# # print(np.array(centroid)-np.array(centroid_new));
	# if(count==8):
	# 	break;
	centroid=centroid_new;
	# print(count)
	
	np.array(clusters).dump("clusters_k_means_pca.dat")
	for i in range(len(clusters)):
		clusters[i]=np.sort(clusters[i]);
	clusters.sort(key=lambda x:min(x))
	np.savetxt("clusters_k_means_pca.txt",clusters,fmt="%s");


print(clusters)
print("Number of iterations "+str(it));
print("Reason 1")