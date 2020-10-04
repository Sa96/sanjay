# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 18:41:09 2020

@author: dell
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data=pd.read_csv("C:\\Users\\dell\\Documents\\excel documents\\games.csv")

print(data.columns)
print(data.shape)

plt.hist(data["average_rating"])

print(data[data["average_rating"]==0].iloc[0])
print(data[data["average_rating"]>0].iloc[0])
data=data[data["users_rated"]>0]
data=data.dropna(axis=0)
plt.hist(data["average_rating"])

corrmat=data.corr()
fig=plt.figure(figsize=(12,9))
sns.heatmap(corrmat,vmax=0.8,square=True)
plt.show()

columns = data.columns.tolist()
columns = [c for c in columns if c not in ["bayes_average_rating", "average_rating", "type", "name", "id"]]
target = "average_rating"

from sklearn.model_selection import train_test_split
train=data.sample(frac=0.8,random_state=1)
test=data.loc[data.index.isin(train.index)]

print(train.shape)
print(test.shape)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(train[columns],train[target])

from sklearn.metrics import mean_squared_error
predictions=model.predict(test[columns])
mean_squared_error(predictions,test[target])

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)
model.fit(train[columns],train[target])
predictions=model.predict(test[columns])
mean_squared_error(predictions,test[target])