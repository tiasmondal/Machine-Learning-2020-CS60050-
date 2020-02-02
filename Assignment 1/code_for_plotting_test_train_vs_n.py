import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
train=[0.09969350888294176,0.09947855184672076,0.012772405961884687,0.007687210876893736,0.01069534034882643,0.014991434095013464,0.017975059286941666,0.012058423070569644,0.008353154123972808]
test=[0.09546810346589259,0.09549645486780316,0.012626258201260896,0.007927011188945017,0.011154379234412265,0.015682916117449475,0.01880301043300922,0.012558210037289439,0.008680118189853002]
n=[1,2,3,4,5,6,7,8,9]
fig=plt.figure()
plt.plot(n,train)
plt.title("training error vs x",color="white")
plt.xlabel("n",color="white")
plt.ylabel("training error",color="white")
plt.show()
fig=plt.figure()
plt.plot(n,test)
plt.title("test error vs x",color="white")
plt.xlabel("n",color="white")
plt.ylabel("test error",color="white")
plt.show()