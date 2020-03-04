import numpy as np
import csv
import pandas as pd
eps = np.finfo(float).eps
from numpy import log2 as log
import pprint
from sklearn.model_selection import train_test_split
from sklearn import metrics

filename = "data_modified_Decision_tree.csv"
data=pd.read_csv(filename,sep=r'\s*,\s*',header=0, encoding='ascii', engine='python')


from sklearn import tree
data=np.array(data);
#print(np.shape(data))
labels=data[:,11:12]

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=0)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)
predictions=clf.predict(x_test)
cm = metrics.confusion_matrix(y_test, predictions)
print("Predictions")
print(predictions)
print("Confusion matrix")
print(cm)
#print(clf)