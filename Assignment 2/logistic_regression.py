import numpy as np
import csv
import pandas as pd
from imblearn.over_sampling import SMOTE

def sigmoid(X):
	return(1/(1+np.exp(-X)));

filename = "data_modified.csv"
data=pd.read_csv(filename,sep=r'\s*,\s*',header=0, encoding='ascii', engine='python')

data=np.array(data);
#print(np.shape(data))
labels=data[:,11:12]
labelset_1=data[0:533,11:12];
labelset_2=data[533:1067,11:12];
labelset_3=data[1067:,11:12];
labels=np.array(labels);
labelset_1=np.array(labelset_1);
labelset_2=np.array(labelset_2);
labelset_3=np.array(labelset_3);
data=data[:,0:11];
dataset_1=data[0:533,0:11];
dataset_2=data[533:1067,0:11];
dataset_3=data[1067:,0:11];
theta=np.random.randn(11);
#print(data);
c_accuracy=0;
c_precision=0;
c_recall=0;

theta=np.resize(theta,(11,1));

l_r=0.001;
error_diff=1;
count=0;
########################################################## 1st fold training ##########################################################
data=np.concatenate((dataset_1,dataset_2),axis=0);
labels=np.concatenate((labelset_1,labelset_2),axis=0);
sm = SMOTE(random_state = 2) 
data, labels = sm.fit_sample(data, labels.ravel()) 

labels=np.resize(labels,(len(labels),1));

while error_diff>=10e-7:
	h=sigmoid(data.dot(theta));
	count=count+1;
	error=-(np.transpose(labels).dot(np.log(h))+(1-np.transpose(labels)).dot(np.log(1-h)))/len(data);
	theta=theta-l_r*(np.transpose(data)).dot(h-labels)/len(data)
	h1=sigmoid(data.dot(theta));
	error1=-(np.transpose(labels).dot(np.log(h1))+(1-np.transpose(labels)).dot(np.log(1-h1)))/len(data);
	
	error_diff=error-error1;
	if(error1>error):
		print("Diverging");
		break;

####################################################### Prediction #######################################################################################
data_test=dataset_3;
labels_target=labelset_3;
h=sigmoid(data_test.dot(theta));
for i in range(0,len(h)):
	if(h[i]>=0.5):
		h[i]=1;
	else:
		h[i]=0;
count1=0;
for i in range(0,len(h)):
	if(h[i]==labels_target[i]):
		count1=count1+1;

print("Number of iterations for 1st fold");
print(count)
print("Accuracy for 1st fold");
print((count1/len(h))*100);
c_accuracy=c_accuracy+ (count1/len(h))*100;
count_precision=0;
count_recall=0;
count_h=0;
count_y=0;
for i in range(0,len(h)):

	if(h[i]==1):
		count_h=count_h+1;
	if(labels_target[i]==1):
		count_y=count_y+1;
for i in range(0,len(h)):
	if(h[i]==1):
		if(labels_target[i]==1):
			count_precision=count_precision+1;
for i in range(0,len(h)):
	if(labels_target[i]==1):
		if(h[i]==1):
			count_recall=count_recall+1;

print("precision for 1st fold");
print(count_precision/count_h);
c_precision=c_precision+count_precision/count_h;
print("recall for 1st fold")
print(count_recall/count_y);
c_recall=c_recall+count_recall/count_y;

################################################################ 2nd fold training #####################################################################3
l_r=0.001;
error_diff=1;
count=0;
theta=np.random.randn(11);
theta=np.resize(theta,(11,1));
########################################################## 2nd fold training ##########################################################
data=np.concatenate((dataset_2,dataset_3),axis=0);
labels=np.concatenate((labelset_2,labelset_3),axis=0);
 
sm = SMOTE(random_state = 2) 
data, labels = sm.fit_sample(data, labels.ravel()) 

labels=np.resize(labels,(len(labels),1));
while error_diff>=10e-7:
	h=sigmoid(data.dot(theta));
	count=count+1;
	error=-(np.transpose(labels).dot(np.log(h))+(1-np.transpose(labels)).dot(np.log(1-h)))/len(data);
	theta=theta-l_r*(np.transpose(data)).dot(h-labels)/len(data)
	h1=sigmoid(data.dot(theta));
	error1=-(np.transpose(labels).dot(np.log(h1))+(1-np.transpose(labels)).dot(np.log(1-h1)))/len(data);
	
	error_diff=error-error1;
	if(error1>error):
		print("Diverging");
		break;

####################################################### Prediction #######################################################################################
data_test=dataset_1;
labels_target=labelset_1;
h=sigmoid(data_test.dot(theta));
for i in range(0,len(h)):
	if(h[i]>=0.5):
		h[i]=1;
	else:
		h[i]=0;
count1=0;
for i in range(0,len(h)):
	if(h[i]==labels_target[i]):
		count1=count1+1;

print("Number of iterations for 2nd fold");
print(count)
print("Accuracy for 2nd fold");
print((count1/len(h))*100);
c_accuracy=c_accuracy+ (count1/len(h))*100;
count_precision=0;
count_recall=0;
count_h=0;
count_y=0;
for i in range(0,len(h)):
	if(h[i]==1):
		count_h=count_h+1;
	if(labels_target[i]==1):
		count_y=count_y+1;
for i in range(0,len(h)):
	if(h[i]==1):
		if(labels_target[i]==1):
			count_precision=count_precision+1;
for i in range(0,len(h)):
	if(labels_target[i]==1):
		if(h[i]==1):
			count_recall=count_recall+1;

print("precision for 2nd fold");
print(count_precision/count_h);
c_precision=c_precision+count_precision/count_h;
print("recall for 2nd fold")
print(count_recall/count_y);
c_recall=c_recall+count_recall/count_y;

################################################################ 3rd fold training ################################################################################

l_r=0.01;
error_diff=1;
count=0;
error1=1;
theta=np.random.randn(11);


theta=np.resize(theta,(11,1));

########################################################## 3nd fold training ##########################################################
data=np.concatenate((dataset_1,dataset_3),axis=0);
labels=np.concatenate((labelset_1,labelset_3),axis=0);

sm = SMOTE(random_state = 2) 
data, labels = sm.fit_sample(data, labels.ravel()) 

labels=np.resize(labels,(len(labels),1));
#exit();

theta=np.resize(theta,(11,1));
while error_diff>=10e-7:
	h=sigmoid(data.dot(theta));
	count=count+1;
	error=-(np.transpose(labels).dot(np.log(h))+(1-np.transpose(labels)).dot(np.log(1-h)))/len(data);
	theta=theta-l_r*(np.transpose(data)).dot(h-labels)/len(data)
	h1=sigmoid(data.dot(theta));
	error1=-(np.transpose(labels).dot(np.log(h1))+(1-np.transpose(labels)).dot(np.log(1-h1)))/len(data);
	
	error_diff=error-error1;
	print(error1);
	if(error1>error):
		print("Diverging");
		break;

####################################################### Prediction #######################################################################################
data_test=dataset_2;
labels_target=labelset_2;
#print(error_diff);
h=sigmoid(data_test.dot(theta));
#print(h)
for i in range(0,len(h)):
	if(h[i]>=0.5):
		h[i]=1;
	else:
		h[i]=0;
count1=0;
for i in range(0,len(h)):
	if(h[i]==labels_target[i]):
		count1=count1+1;

print("Number of iterations for 3rd fold");
print(count)
print("Accuracy for 3rd fold");
print((count1/len(h))*100)
c_accuracy=c_accuracy+ (count1/len(h))*100;
count_precision=0;
count_recall=0;
count_h=0;
count_y=0;
for i in range(0,len(h)):
	if(h[i]==1):
		count_h=count_h+1;
	if(labels_target[i]==1):
		count_y=count_y+1;
for i in range(0,len(h)):
	if(h[i]==1):
		if(labels_target[i]==1):
			count_precision=count_precision+1;
for i in range(0,len(h)):
	if(labels_target[i]==1):
		if(h[i]==1):
			count_recall=count_recall+1;

print("precision for 3rd fold");
print(count_precision/count_h);
c_precision=c_precision+count_precision/count_h;
print("recall for 3rd fold")
print(count_recall/count_y);
c_recall=c_recall+count_recall/count_y;
print("Average accuracy")
print(c_accuracy/3)
print("Average Precision")
print(c_precision/3)
print("Average Recall")
print(c_recall/3)


