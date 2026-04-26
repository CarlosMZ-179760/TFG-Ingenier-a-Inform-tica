# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 13:11:12 2026

@author: TESTER
"""

from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import random
import numpy as np
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import TimeSeriesSplit
from six import StringIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import CategoricalNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier


train=0.8 
test=1-train
nColsLearn=6
#1.Distribuye todos los parámetros de manera uniforme entre entrenamiento y test: la semilla se usa para decidir qué csv's se usan para entrenar y cuáles para probar. Por ejemplo: semillas 1-24 para entrenar, 25-30 de test.
#2. Utiliza las instancias de cierto tipo como entrenamiento, y otras de test

dataDirectory=Path("Respuestas/StreamsSmall")
filePathMask='**/*.csv'
dataIterator=dataDirectory.glob(filePathMask)

i=0
buffer=pd.DataFrame(data=None)

total=0

for file in dataIterator:
    nColsProv = len(open(file).readline().split(','))
    if nColsProv!=nColsLearn+2:
        print(nColsProv, nColsLearn)
        raise Exception("Incorrect number of streams")
    temp=pd.read_csv(file, usecols=range(1,nColsProv), dtype="int8")
    total+=temp.shape[0]
Y=np.empty(shape=[total])
X=np.empty(shape=[total,nColsLearn])
dataIterator=dataDirectory.glob(filePathMask)
k=0
for file in dataIterator:
    temp=pd.read_csv(file, usecols=range(1,nColsLearn+2))
    fraction=temp.to_numpy(dtype="int64")
    yFraction=fraction[:,-1]
    #print(yFraction)
    xFraction=fraction[:,0:-1]
    #print(xFraction.shape, xFraction)
    #yFraction=temp["Ground-truth"].to_numpy(dtype="int64")
    #print(yFraction.shape)
    for j in range(yFraction.shape[0]):
        #print(yFraction[j])
        #print(xFraction[j])
        #print(total,k)
        Y[k]=yFraction[j]
        X[k]=xFraction[j]
        #print(Y[k], X[k])
        k+=1
    #Y=np.append(Y,yFraction)
    #X=np.append(X,xFraction)
    #print(Y)
    #buffer=buffer.append(temp, ignore_index=True)
    i+=1
    #print(i)
    #fileName=file.name.removesuffix('.csv')
    #print(fileName)
    #regex = re.compile(r'\d+')
    #seed=int(regex.findall(fileName)[0])
    #print(int(seed))
    # if(seed<25):
    #     #Entrenamiento
    # else:
    #     #Archivo de test
        
    # if(random.random()<=train):
    #     #Archivo de entrenamiento
    # else:
    #     #Archivo de test        
print(X.shape, X)
print(Y.shape, Y)

tscv=100#TimeSeriesSplit(n_splits=100)
score='accuracy'
neighs=5
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test, random_state=0)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
clf = DecisionTreeClassifier(random_state=0)
print("CV=",tscv)
print("Directory: ",dataDirectory)
print("Vecinos= ", neighs)
print("Score= ", score)
print("Decision Tree")
score_dt=cross_val_score(clf, X, Y, cv=tscv, scoring=score)
clf.fit(X_train, y_train)
print(#score_dt, 
      score_dt.mean(), score_dt.std())
plt.hist(score_dt, bins=20)
plt.title("Árbol de decisión")
plt.show()
gnb = CategoricalNB()
print("Naive Bayes")
score_nb=cross_val_score(gnb, X, Y, cv=tscv, scoring=score)
print(#score_nb, 
      score_nb.mean(), score_nb.std())
plt.hist(score_nb, bins=20)
plt.title("NB")
plt.show()
# dot_data = StringIO()
# tree.export_graphviz(clf, out_file = dot_data,  
#                 filled = True, rounded = True,
#                 special_characters = True)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
# Image(graph.create_png())
print(tree.export_text(clf))
print(clf)
print(gnb)
#y_pred = gnb.fit(X_train, y_train).predict(X_test_scaled)
#print("Number of mislabeled points out of a total %d points : %d"
#      % (X_test_scaled.shape[0], (y_test != y_pred).sum()))
#print(cohen_kappa_score(y_pred, y_test))
#KNN=KNeighborsClassifier(neighs)
#print("KNN")
#score_knn=cross_val_score(KNN, X, Y, cv=tscv, scoring=score)
#plt.hist(score_knn, bins=20)
#plt.title("KNN")
#plt.show()
#print(#score_knn, 
#      score_knn.mean(), score_knn.std())
#svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
#print("SVM")
#score_svm=cross_val_score(svm_classifier, X, Y, cv=tscv, scoring='f1')
#print(score_svm, score_svm.mean(), score_svm.std())

