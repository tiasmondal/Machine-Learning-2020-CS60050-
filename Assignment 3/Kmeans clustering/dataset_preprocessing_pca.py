import numpy as np
import sys
import csv
import pandas as pd
from tqdm import tqdm


filename = "AllBooks_baseline_DTM_Labelled_pca.csv"
pd.options.mode.chained_assignment = None
with pd.option_context('display.precision', 10):
	data=pd.read_csv(filename,sep=r'\s*,\s*',header=0, encoding='ascii', engine='python')


# print(data["Religious_texts"])

headers=data.columns.values;
for i in range(0,len(data["Religious_texts"])):
	data["Religious_texts"][i]=data["Religious_texts"][i].split("_")[0]
#print(data["Religious_texts"].unique())


#print(data["Religious_texts"].value_counts())
headers=headers[2:]


print(headers)
print(len(headers))
idf=np.zeros(len(headers),)
for i in range(0,len(headers)):
	count=0;
	for j in range(0,len(data['Religious_texts'])):
		if(data[headers[i]][j]>=1):
			count=count+1;
	idf[i]=np.log(1+len(data['Religious_texts'])/(1+count))/np.log(np.exp(1));
	
print("idf");
print(idf);
idf=np.array(idf)
data1=np.array(data);
data1=data1[:,2:]

# norm=np.zeros(len(data['Religious_texts']),);
# for i in range(0,len(data['Religious_texts'])):
# 	normsum=np.sum(data1[i]**2);
# 	normsum=normsum**0.5;
# 	norm[i]=normsum;

# print("Norm");
# print(norm)
data2=data
# for i in range(0,len(headers)):
# 	for j in range(0,len(data['Religious_texts'])):
# 		# if(j==0  and i==15):
# 		# 	print((data[headers[i]][j]*idf[i])/norm[j])
# 		# 	print(headers[i])
# 		# 	print(idf[i])
# 		# 	print(norm[j])
# 		x=(data[headers[i]][j]*idf[i])/norm[j];
# 		# if(j==0  and i==15):
# 		# 	print(x)
# 		data[headers[i]][j]=x;


# 		# if(j==0  and i==15):
# 		# 	print("yup")
# 		# 	print(x)
# 		# 	data[headers[i]][j]=x;
# 		# 	print(data[headers[i]][j]);
			
# 	print(i)
count=0;
if(np.shape(data1)[0]!=len(data['Religious_texts']) or np.shape(data1)[1]!=len(headers)):
	print(np.shape(data1)[0]);
	print(len(data['Religious_texts']));
	print(np.shape(data1)[1]);
	print(len(headers));
	

for i in tqdm(range(0,len(data['Religious_texts']))):
	for j in tqdm(range(0,len(headers))):
		data1[i][j]=(data1[i][j]*idf[j]);
		

norm=np.zeros(len(data['Religious_texts']),);
for i in range(0,len(data['Religious_texts'])):
	normsum=np.sum(data1[i]**2);
	normsum=normsum**0.5;
	norm[i]=normsum;

print("Norm");
print(norm)

for i in tqdm(range(0,len(data['Religious_texts']))):
	for j in tqdm(range(0,len(headers))):

		if(norm[i]!=0):
			data1[i][j]=(data1[i][j])/norm[i];
		else:
			data1[i][j]=(data1[i][j]);


# data1[13][0]=1;

		
	
	

# print(data1)

# for i in range(0,np.shape(data1)[0]):
# 	for j in range(0,np.shape(data1)[1]):
# 		if(data1[i][j]!=0):
# 			print(data1[i][j])

dict_texts={};
dict_texts["Religious_texts"]=data['Religious_texts'];
for i in range(0,len(headers)):
	dict_texts[headers[i]]=data1[:,i]


dataset=pd.DataFrame(dict_texts);

print(dataset)
# data=np.array(data);
# for i in range(0,len(data["Religious_texts"])):
# 	for j in range(0,8267):
# 		data[i][j]=log(1+len(data["Religious_texts"]))/()

df = pd.DataFrame(dataset); 
df.to_csv("data_modified_pca.csv");