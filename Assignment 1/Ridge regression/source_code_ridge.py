import numpy as np
import csv
import pandas as pd
filename = "train.csv"
data=pd.read_csv(filename,sep=r'\s*,\s*',header=0, encoding='ascii', engine='python')
feature=np.ones((1000,1))               #Features
order=1                             #Order of the polynomial that can be changed
labels=np.array(data["Label"])        #Labels
labels=labels.reshape((1000,1))
l_r=0.05
arrerror=[]
countarr=[]
for num in range(1,order+1):
  x=np.array(data["Feature"])
  x=x.reshape((1000,1))
  feature=np.append(feature,x**num,axis=1)
weights=np.zeros((order+1,1))                        #weights
error_diff=10000000
error=10000
count=0
lambda1=0.5                                        #lambda value of the regression that can be changed
while (error_diff>=0.0000001):                                 #error difference threshold set to 10^-7
  error1=np.sum((feature.dot(weights)-labels)**2)/(2*1000)+lambda1*np.sum(weights**2)    #Previous error
  weights=weights-l_r*(np.transpose(feature)).dot(feature.dot(weights)-labels)/1000-(2*l_r*lambda1)*weights  #weights update
  error=np.sum((feature.dot(weights)-labels)**2)/(2*1000)+lambda1*np.sum(weights**2)   #Current errpr
  arrerror=np.append(arrerror,error)              #Estimating learning curve
  countarr=np.append(countarr,count)
  if(error>error1):
    print("Diverging")
    break;
  count=count+1
  error_diff=abs(error1-error)
print("Training error")
print(error)
print("Total number of iterations")
print(count)
#################################################### Plotting the test error ###############################################
import matplotlib.pyplot as plt
filename = "test.csv"
data=pd.read_csv(filename,sep=r'\s*,\s*',header=0, encoding='ascii', engine='python')

X = data["Feature"]
X=np.array(X)
Y = data["Label"]
Y=np.array(Y)
Y=Y.reshape((200,1))
feature=np.array(data["Feature"])
feature=np.ones((200,1))
for num in range(1,order+1):
  x=np.array(data["Feature"])
  x=x.reshape((200,1))
  feature=np.append(feature,x**num,axis=1)

fig=plt.figure()
plt.scatter(X,Y,label="Actual test data")

Z=feature.dot(weights)
err=np.sum((Z-Y)**2)/400
print("Test error")
print(err)
Z=Z.reshape((200,))
Z=np.array(Z)
print("Plot on test data")
plt.scatter(X,Z,label="Predicted test data")
plt.xlabel("feature points",color="white")
plt.ylabel("Predicted labels",color="white")
plt.title("Fitted curve for order "+ str(order)+" and lambda "+str(lambda1)+" on test data for ridge regression",color="white")
plt.legend()
plt.show()
fig=plt.figure()
plt.scatter(countarr,arrerror)
plt.xlabel("Number of iterations",color="white")
plt.ylabel("Mean squared error",color="white")
plt.show()
################################################# Plotting the training error ##################################
import matplotlib.pyplot as plt
filename = "train.csv"
data=pd.read_csv(filename,sep=r'\s*,\s*',header=0, encoding='ascii', engine='python')

X = data["Feature"] 
X=np.array(X)
Y = data["Label"] 
Y=np.array(Y)
Y=Y.reshape((1000,1))
feature=np.array(data["Feature"])
feature=np.ones((1000,1))
for num in range(1,order+1):
  x=np.array(data["Feature"])
  x=x.reshape((1000,1))
  feature=np.append(feature,x**num,axis=1)

fig=plt.figure()
plt.scatter(X,Y,label="Actual training data")
Z=feature.dot(weights)
err=np.sum((Z-Y)**2)/2000

Z=Z.reshape((1000,))
Z=np.array(Z)
print("Plot on training data")
plt.scatter(X,Z,label="Predicted training data")
plt.xlabel("feature points",color="white")
plt.ylabel("Predicted labels",color="white")
plt.legend()
plt.title("Fitted curve for order "+str(order)+" and lambda "+str(lambda1)+" on training data for ridge regression",color="white")
plt.show()


  

