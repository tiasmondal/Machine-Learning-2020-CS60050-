import numpy as np
import csv
import pandas as pd
eps = np.finfo(float).eps
from numpy import log2 as log
import pprint

filename = "data_modified_Decision_tree.csv"
data=pd.read_csv(filename,sep=r'\s*,\s*',header=0, encoding='ascii', engine='python')

print(data)
from sklearn import tree
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
print(clf.predict([[2., 2.]]))
print(clf)