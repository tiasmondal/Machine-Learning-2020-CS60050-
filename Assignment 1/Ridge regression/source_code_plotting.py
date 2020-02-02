import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
filename = "train.csv"
data=pd.read_csv(filename,sep=r'\s*,\s*',header=0, encoding='ascii', engine='python')

X = data["Feature"]
X=np.array(X)
Y = data["Label"]
Y=np.array(Y)
Y=Y.reshape((1000,1))


fig=plt.figure()
plt.scatter(X,Y,label="Actual test data")
plt.show()