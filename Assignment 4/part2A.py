import numpy as np
import random
import sys
import csv
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from decimal import *
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore");
mean=np.zeros(7,);
squared=np.zeros(7,);
sd=np.zeros(7,);
def preprocess(data):
	train=np.zeros((168,8));
	test=np.zeros((32,8));
	for i in range(0,7):
		mean[i]=np.sum(data[:,i])/210;
		sd[i]=(np.sum((data[:,i]-mean[i])**2)/210)**0.5;
		data[:,i]=(data[:,i]-mean[i])/sd[i];
	return data;
filename = "dataset.csv"
#print("Loading Dataset..................\n")
print("Part 2A\n")
pd.options.mode.chained_assignment = None
with pd.option_context('display.precision', 10):
	data=pd.read_csv(filename,sep=r'\s*,\s*',header=0, encoding='ascii', engine='python')
data=np.array(data);
data=preprocess(data)
feature_data=data[:,0:7]
label_data=data[:,7:8]
label_data1=[];
#print(label_data)
for i in range(0,210):
	x=[0,0,0]
	x[int(label_data[i])-1]=1;
	label_data1.append(x)
X_train, X_test = feature_data[:192], feature_data[192:]
y_train, y_test = label_data1[:192], label_data1[192:]
mlp = MLPClassifier(hidden_layer_sizes=(32,),activation='logistic', max_iter=200, alpha=0,
                    solver='sgd', verbose=10, random_state=1,
                    learning_rate_init=.01)
mlp.fit(X_train, y_train)
a=mlp.score(X_train, y_train)
b=mlp.score(X_test, y_test)
print("\nTraining Accuracy: "+ str(float(a)*100))
print("\nTest Accuracy: "+ str(float(b)*100))


