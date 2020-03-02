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
  if(data["quality"][i]<=6):
    data["quality"][i]=0;
    count0=count0+1;
  else:
    data["quality"][i]=1;
    count1=count1+1;

for i in range(0,12):
  for j in range(0,len(data["quality"])):
    data[headers[i]][j]=(data[headers[i]][j]-min(data[headers[i]]))/(max(data[headers[i]])-min(data[headers[i]]));

df = pd.DataFrame(data); 
df.to_csv("data_modified.csv");
print(count0);
print(count1);