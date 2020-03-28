import numpy as np
import csv
import pandas as pd
from tqdm import tqdm
import math
import random

print("Which method you want to find NMI for K_means clustering? For original dataset press 1 or for PCA press 2");
method=input();
method=int(method);
if(method==1):
	filename = "AllBooks_baseline_DTM_Labelled.csv"
	clusters = np.load("clusters_k_means.dat", allow_pickle=True)
else:
	filename = "AllBooks_baseline_DTM_Labelled_pca.csv"
	clusters = np.load("clusters_k_means_pca.dat", allow_pickle=True)

pd.options.mode.chained_assignment = None
with pd.option_context('display.precision', 10):
	data=pd.read_csv(filename,sep=r'\s*,\s*',header=0, encoding='ascii', engine='python')
headers=data.columns.values;
for i in range(0,len(data["Religious_texts"])):
	data["Religious_texts"][i]=data["Religious_texts"][i].split("_")[0]

labels=data['Religious_texts'].unique()
classes=[[],[],[],[],[],[],[],[]]
#for i in range(0,len(labels)):
i=0;
for j in range(0,len(labels)):
	count=0;
	for i in range(0,len(data['Religious_texts'])):
		if(data['Religious_texts'][i]==labels[j]):
			classes[j].append(i);
total_len=0;
p_class=[];
h_class=0;
for i in range(0,8):
	total_len+=len(classes[i])

#print(classes)
for i in range(0,8):
	p_class.append(len(classes[i])/total_len);
	h_class+=-(len(classes[i])/total_len)*(np.log(len(classes[i])/total_len)/np.log(2))
print("Probability of P(Y=1),P(Y=2).......")
print(p_class)
print("\n")
print("Entropy of H(Y=1),H(Y=2)..........")
print(h_class)
print("\n")

total_len=0;
p_clusters=[];
for i in range(0,8):
	total_len+=len(clusters[i])
h_clusters=0;
for i in range(0,8):
	p_clusters.append(len(clusters[i])/total_len);
	h_clusters+=-(len(clusters[i])/total_len)*(np.log(len(clusters[i])/total_len)/np.log(2))
print("Probability of P(C=1),P(C=2).......")
print(p_clusters)
print("\n")
print("Entropy of H(C=1),H(C=2)..........")
print(h_clusters)
print("\n")
p_y_c=[[],[],[],[],[],[],[],[]];
for t in range(0,8):
	for i in range(0,8):
		count=0;
		for j in range(0,len(classes[i])):
			for k in range(0,len(clusters[t])):
				if(clusters[t][k]==classes[i][j]):
					count=count+1;
				#if(i==0):
					#print(clusters[i][j])
		p_y_c[t].append(count/len(clusters[t]))
print("Probability of [P(Y=1|C=1),P(Y=2|C=1)..........], [P(Y=1|C=2),P(Y=2|C=2)..........], [P(Y=1|C=3),P(Y=2|C=3)..........]")
print(p_y_c)
print("\n")
h_y_c=[0,0,0,0,0,0,0,0]
for i in range(0,8):
	for j in range(0,len(p_y_c[i])):
		if(p_y_c[i][j]!=0):
			h_y_c[i]+=p_clusters[i]*(-p_y_c[i][j]*np.log(p_y_c[i][j])/np.log(2))

print("Entropy of H(Y|C=1), H(Y|C=2), H(Y|C=3), H(Y|C=4).....................")
print(h_y_c)
print("\n")
i_y_c=0;

i_y_c=h_class-np.sum(np.array(h_y_c));
print("Mutual Information = "+str(i_y_c))
print("\n")
print("NMI = "+str((2*i_y_c)/(h_class+h_clusters)))






