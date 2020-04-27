#Good
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
random.seed(1);
getcontext().prec=30
fig=plt.figure();
ax1=fig.add_subplot(1,1,1)
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
	train_data=data[0:168,:7]
	#print(np.shape(train_data))
	test_data=data[168:,:7]
	train_label=data[0:168,7:8]
	test_label=data[168:,7:8]

	return train_data,train_label,test_data,test_label
	
minibatch=[]
minibatch_labels=[];

def data_loader(data,label):
	count=0;
	k=0;
	for i in range(0,5):
		minibatch.append(data[k:k+32,:])
		minibatch_labels.append(label[k:k+32,:])
		k=k+32
		count=count+1;
	#print("Number itr "+str(count))
	minibatch.append(data[160:,:])
	minibatch_labels.append(label[160:,:])

def weight_initializer():
	w1=2*np.random.rand(64,8)-1;
	w2=2*np.random.rand(32,65)-1;
	w3=2*np.random.rand(3,33)-1;
	return w1,w2,w3;
def softmax(arr):
	#print(arr)
	tot=float(format(np.sum(np.exp(arr)),'.20f'))
	# y=np.exp(arr)/(tot);
	# for i in range(0,3):
	# 	if(y[i]==1):
	# 		print(arr);
	# 		exit();
	return(np.exp(arr)/(tot))
def forward_pass(weight,x,status):
	x=np.insert(x,0,1,axis=0);
	if(status=='hidden'):
		#return(1/(1+np.exp(-weight.dot(x))))
		mat=weight.dot(x);
		#print(mat)
		for i in range(0,mat.shape[0]):
			if(mat[i][0]<0):
				mat[i][0]=0;

		return(mat)
	else:
		return(softmax(weight.dot(x)))
def derivative(mat):
	for i in range(0,mat.shape[0]):
		if(mat[i][0]<=0):
			mat[i][0]=0;
		else:
			mat[i][0]=1;
	return(mat);

def back_propag(w1,w2,w3,input1,out1,out2,out3,label,alpha):
	#input1=input1.insert(input1,0,1,axis=0)
	#delta2=-label*np.log(out2)-(1-label)*np.log(1-out2);
	delta3=-(label-out3);
	delta2=derivative(out2)*np.transpose(w3).dot(delta3);
	delta2=delta2[1:,:]
	delta1=derivative(out1)*np.transpose(w2).dot(delta2);
	#delta1=delta1[1:,:]
	#w2=w2-alpha*(out1.dot(np.reshape(delta2,(1,delta2.shape[0]))))
	#w1=w1-alpha*(input1.dot(np.reshape(delta1,(1,delta1.shape[0]))))
	return delta1,delta2,delta3

def oneinsert(array):
	array=np.insert(array,1,0,axis=0)
	array=array.reshape(array.shape[0],1)
	return array;
def one_hot(n):
	x=np.zeros((3,1));
	#print(int(n[0]));
	x[int(n[0])-1][0]=1;
	return x;
def train_batch(minibatch,minibatch_labels,w1,w2,w3,alpha,test_data,test_label,train_data,train_label):
	prev_error=1000000;
	tr=[];
	te=[];
	x=[];
	prevw1=w1;
	prevw2=w2;
	prevw3=w3;
	for epochs in range(0,200):
		error=0;
		cap_delta0=0;
		cap_delta1=0;
		cap_delta2=0;
		for j in range (0,6):
			if(j==5):
				k=8;
			else:
				k=32;
			for i in range(0,k):
				wl=forward_pass(w1,np.reshape(minibatch[j][i],(7,1)),'hidden') # First forward pass
				out=forward_pass(w2,wl,'hidden')   # 2nd Forward pass
				out1=forward_pass(w3,out,'final')  #Third Forward Pass
				#print(out1);



			#out=out.reshape((out.shape[1],1))
			#print(out)
			#wl=wl.reshape((wl.shape[1],1))
#minibatch_labels[0][0]=np.array([1,0,0]);
				wl=np.insert(wl,0,1,axis=0)
				out=np.insert(out,0,1,axis=0)
#minibatch[0][0]=np.insert(minibatch[0][0],1,0,axis=0)
				one_hot_vec=one_hot(minibatch_labels[j][i])
				input1=oneinsert(minibatch[j][i])
			#print(np.shape(w1),np.shape(w2),np.shape(input1),np.shape(wl),np.shape(out),np.shape(one_hot_vec))
				#print(out1);
				# for tm in range(0,3):
				# 	if(out1[tm][0]==0 or out1[tm][0]==1):
				# 		print(out1)
				# 		exit();
				error+=(np.sum(-one_hot_vec*np.log(out1+0.001)-(1-one_hot_vec)*np.log(1-out1+0.001)));
			#error+=np.sum((one_hot_vec-out)**2)
				delta1,delta2,delta3=back_propag(w1,w2,w3,input1,wl,out,out1,one_hot_vec,0.001)
			
			#wl=wl[1:,:]
			#print(np.shape(delta1),np.shape(input1))
				delta1=delta1[1:,:];
				#delta2=delta2[1:,:];
				cap_delta2=cap_delta2+delta3.dot(np.transpose(out));
				cap_delta1=cap_delta1+delta2.dot(np.transpose(wl));
				cap_delta0=cap_delta0+delta1.dot(np.transpose(input1));
			#print(np.shape(cap_delta0),np.shape(w1))
			#print(np.shape(w1),np.shape(w2))

			

			w2=w2-alpha*cap_delta1/k;
			w1=w1-alpha*cap_delta0/k;
			w3=w3-alpha*cap_delta2/k;
		print(error)
		if(error>prev_error):
				#wx1,wx2,wx3=train_batch(minibatch,minibatch_labels,prevw1,prevw2,prevw3,alpha,test_data,test_label,train_data,train_label)
				#return wx1,wx2,wx3
				break;
		if((epochs+1)%10==0):
			train=predict(train_data,train_label,w1,w2,w3)
			test=predict(test_data,test_label,w1,w2,w3)
			x.append((epochs+1)/10)
			tr.append(train)
			te.append(test);
			
			print("Train acc:= "+str(train)+" Test acc:= "+str(test))
			ax1.clear();
			ax1.plot(x,tr,label='Training accuracy');
			ax1.plot(x,te,label='Test Accuracy');
			ax1.legend()
			plt.pause(0.05)
		
		prev_error=error;
		
		# if(error>prev_error):
		# 	break;
		# else:
		# 	prev_error=error

		#print(error);
	plt.show();
	print("\nTraining complete...........\n")
	return w1,w2,w3

def predict(data,label,w1,w2,w3):
	correct=0;
	for i in range(0,data.shape[0]):
		wl=forward_pass(w1,np.reshape(data[i],(7,1)),'hidden') # First forward pass
		out=forward_pass(w2,wl,'hidden')   # 2nd Forward pass
		out1=forward_pass(w3,out,'final')  #Third Forward Pass
		#print("Got "+str(np.argmax(out)+1)+" Expected "+str(label[i]))
		if(np.argmax(out1)+1==label[i]):
			correct=correct+1;
	return((correct*100)/data.shape[0])






filename = "dataset.csv"
print("Loading Dataset..................\n")
pd.options.mode.chained_assignment = None
with pd.option_context('display.precision', 10):
	data=pd.read_csv(filename,sep=r'\s*,\s*',header=0, encoding='ascii', engine='python')
data=np.array(data);
print("Preprocessing Data.....................\n")
train_data,train_label,test_data,test_label=preprocess(data);
print("Splitting into train and test...........\n")
data_loader(train_data,train_label);
print("Initializing initial weights.............\n")
w1,w2,w3=weight_initializer();
#print(minibatch[4][0]);
#wl=forward_pass(w1,np.reshape(minibatch[0][0],(7,1)),'hidden') 
#print(np.shape(test_label))
#exit();
print("Traning using forward pass and backpropag.......\n")
w1,w2,w3=train_batch(minibatch,minibatch_labels,w1,w2,w3,0.01,test_data,test_label,train_data,train_label)
print("Predicting...........\n")
train_acc=predict(train_data,train_label,w1,w2,w3)
test_acc=predict(test_data,test_label,w1,w2,w3)
print("Final Train acc:= "+str(train_acc)+" Final Test acc:= "+str(test_acc))
exit();
print(forward_pass(w1,np.reshape(minibatch[0][0],(1,7)),'hidden'))
wl=forward_pass(w1,np.reshape(minibatch[0][0],(1,7)),'hidden') # First forward pass
out=forward_pass(w2,wl,'final')   # 2nd Forward pass
out=out.reshape((out.shape[1],1))
wl=wl.reshape((wl.shape[1],1))
#minibatch_labels[0][0]=np.array([1,0,0]);
wl=np.insert(wl,0,1,axis=0)
#minibatch[0][0]=np.insert(minibatch[0][0],1,0,axis=0)
one_hot_vec=one_hot(minibatch_labels[0][0])
input1=oneinsert(minibatch[0][0])
print(np.shape(w1),np.shape(w2),np.shape(input1),np.shape(wl),np.shape(out),np.shape(one_hot_vec))
w1,w2=back_propag(w1,w2,input1,wl,out,one_hot_vec,0.1)



