import numpy as np
import csv
import pandas as pd
filename = "winequality-red.csv"
data=pd.read_csv(filename,sep=';',header=0, encoding='ascii', engine='python')



headers=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"];

#for i in headers:
  #print(data[i])
count0=0;
count1=0;
for i in range(0,len(data["quality"])):
  if(data["quality"][i]<5):
    data["quality"][i]=0;
    
  elif(data["quality"][i]==5 or data["quality"][i]==6):
    data["quality"][i]=1;
  else:
  	data["quality"][i]=2;


for i in range(0,11):
	mean=0;
	count0=0;
	count1=0;
	count2=0;
	count3=0;
	sigma=0;
	for k in range(0,len(data["quality"])):
		sigma=sigma+(data[headers[i]][k]**2)*data["quality"][k]
		mean=mean+data[headers[i]][k]
	mean=mean/len(data["quality"]);
	sigma=sigma**0.5;
	for j in range(0,len(data["quality"])):
		data[headers[i]][j]=(data[headers[i]][j]-mean)/sigma;
	a=(max(data[headers[i]])-min(data[headers[i]]))/4;
	for l in range(0,len(data["quality"])):
		if(data[headers[i]][l]>=min(data[headers[i]]) and data[headers[i]][l]< min(data[headers[i]])+a):
			data[headers[i]][l]=0;
			count0=count0+1;
		elif(data[headers[i]][l]>=min(data[headers[i]])+a and data[headers[i]][l]< min(data[headers[i]])+2*a):
			data[headers[i]][l]=1;
			count1=count1+1;
		elif(data[headers[i]][l]>=min(data[headers[i]])+2*a and data[headers[i]][l]< min(data[headers[i]])+3*a):
			data[headers[i]][l]=2;
			count2=count2+1;
		else:
			data[headers[i]][l]=3;
			count3=count3+1;
	print(i);
	print(count0);
	print(count1);
	print(count2);
	print(count3);
print(data);
df = pd.DataFrame(data); 
df.to_csv("data_modified_Decision_Tree.csv");
