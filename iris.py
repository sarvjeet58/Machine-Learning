import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
import seaborn as sns

from sklearn.datasets import load_iris
dataset = load_iris()

X = dataset.data
y = dataset.target

plt.scatter(X[y==0,0],X[y==0,1],c='r',label='setosa')
plt.scatter(X[y==1,0],X[y==1,1],c='b',label='versicolor')
plt.scatter(X[y==2,0],X[y==2,1],c='g',label='verginica')
plt.xlabel('sepal_length')
plt.ylabel('sepal_width')
plt.legend()
plt.show()

plt.scatter(X[y==0,2],X[y==0,3],c='r',label='setosa')
plt.scatter(X[y==1,2],X[y==1,3],c='b',label='versicolor')
plt.scatter(X[y==2,2],X[y==2,3],c='g',label='verginica')
plt.legend()
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)

y_pred = log_reg.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

log_reg.score(X_train,y_train)
log_reg.score(X_test,y_test)
log_reg.score(X,y)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train, y_train)

knn.score(X_train, y_train)
knn.score(X_test, y_test)

from sklearn.metrics import precision_score, recall_score, f1_score
precision_score(y_test,y_pred,average = 'micro')
recall_score(y_test,y_pred,average = 'micro')
f1_score(y_test,y_pred,average = 'micro')

from sklearn.svm import SVC
svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)
svm.fit(X_train, y_train)

svm.score(X_train, y_train)
svm.score(X_test, y_test)

from sklearn.tree import DecisionTreeClassifier
dtc= DecisionTreeClassifier()
dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)
metrics.accuracy_score(y_pred,y_test)









