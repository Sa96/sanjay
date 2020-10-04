# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 09:19:24 2020

@author: dell
"""

import sys
import pandas
import numpy
import matplotlib
import seaborn
import scipy

print('python :{}'.format(sys.version))
print('pandas :{}'.format(pandas.__version__))
print('numpy :{}'.format(numpy.__version__))
print('matplotlib :{}'.format(matplotlib.__version__))
print('seaborn :{}'.format(seaborn.__version__))
print('scipy :{}'.format(scipy.__version__))

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

dataset=pd.read_csv('C:\\Users\\dell\\Documents\\excel documents\\creditcard.csv')
print(dataset.columns)

dataset=dataset.sample(frac=0.1, random_state=1)
print(dataset.shape)
print(dataset.describe())

dataset.hist(figsize=(20,20))
plt.show()


Fraud = dataset[dataset['Class'] == 1]
Valid = dataset[dataset['Class'] == 0]

outlier_fraction = len(Fraud)/float(len(Valid))
print(outlier_fraction)

print('Fraud Cases: {}'.format(len(dataset[dataset['Class'] == 1])))
print('Valid Transactions: {}'.format(len(dataset[dataset['Class'] == 0])))

corrmat=dataset.corr()
fig = plt.figure(figsize = (12, 9))

sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()

columns = dataset.columns.tolist()

columns = [c for c in columns if c not in ["Class"]]

target = "Class"

X = dataset[columns]
Y = dataset[target]

print(X.shape)
print(Y.shape)

from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

state=1

classifiers=classifiers = {"Isolation Forest": IsolationForest(max_samples=len(X),
                                        contamination=outlier_fraction,
                                        random_state=state),
                           "Local Outlier Factor": LocalOutlierFactor(
                            n_neighbors=20,
                            contamination=outlier_fraction)}

plt.figure(figsize=(9, 7))
n_outliers = len(Fraud)


for i, (clf_name, clf) in enumerate(classifiers.items()):
    
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
     
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    
    n_errors = (y_pred != Y).sum()
    
    # Run classification metrics
    print('{}: {}'.format(clf_name, n_errors))
    print(accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred))
