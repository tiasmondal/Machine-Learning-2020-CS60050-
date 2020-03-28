import numpy as np
import sys
import csv
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA


filename = "AllBooks_baseline_DTM_Labelled.csv"
pd.options.mode.chained_assignment = None
with pd.option_context('display.precision', 10):
	data=pd.read_csv(filename,sep=r'\s*,\s*',header=0, encoding='ascii', engine='python')


# print(data["Religious_texts"])

headers=data.columns.values;
for i in range(0,len(data["Religious_texts"])):
	data["Religious_texts"][i]=data["Religious_texts"][i].split("_")[0]
#print(data["Religious_texts"].unique())


#print(data["Religious_texts"].value_counts())
headers=headers[1:]


print(headers)
print(len(headers))
data1=np.array(data);
data1=data1[:,1:]
print("Normalizing vectors, Please wait.......")
for i in tqdm(range(0,np.shape(data1)[0])):
	data1[i]=data1[i]/np.sqrt(np.sum(data1**2))
pca=PCA(n_components=100);
pca.fit(data1);
data1=pca.transform(data1)


if(np.shape(data1)[0]!=len(data['Religious_texts'])):
	print("START")
	print(np.shape(data1)[0]);
	print(len(data['Religious_texts']));
	print(np.shape(data1)[1]);
	print(len(headers));
	print("IN THIS")
	exit();


dict_texts={};
dict_texts["Religious_texts"]=data['Religious_texts'];
for i in range(0,np.shape(data1)[1]):
	dict_texts[headers[i]]=data1[:,i]


dataset=pd.DataFrame(dict_texts);

print(dataset)
# data=np.array(data);
# for i in range(0,len(data["Religious_texts"])):
# 	for j in range(0,8267):
# 		data[i][j]=log(1+len(data["Religious_texts"]))/()

df = pd.DataFrame(dataset); 
df.to_csv("AllBooks_baseline_DTM_Labelled_pca.csv");