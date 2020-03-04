import numpy as np
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

filename = "data_modified.csv"
data=pd.read_csv(filename,sep=r'\s*,\s*',header=0, encoding='ascii', engine='python')

data=np.array(data);
#print(np.shape(data))
labels=data[:,11:12]
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=0)
logisticRegr = LogisticRegression(solver='saga',penalty='none');

logisticRegr.fit(x_train, y_train)
x=logisticRegr.predict(x_test[0].reshape(1,-1));
score = logisticRegr.score(x_test, y_test)
print("Accuracy")
print(score)
predictions = logisticRegr.predict(x_test)
cm = metrics.confusion_matrix(y_test, predictions)

precision=cm[0][0]/(cm[0][0]+cm[0][1]);
recall = cm[0][0]/(cm[0][0]+cm[1][0]);
print("Precision")
print(precision);
print("Recall");
print(recall)
print("Confusion matrix")
print(cm)
