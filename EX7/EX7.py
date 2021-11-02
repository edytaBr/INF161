#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 11:40:27 2021

@author: edyta
"""
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

ratings = pd.read_csv('ratings.csv', decimal=',')
item = pd.read_csv('item_based.csv', decimal=',')
coll = ratings - ratings.mean()
coll.drop(['Unnamed: 0'], axis=1, inplace=True)

corr = round(coll.corr(method = 'pearson'), 2)
collaborative = pd.read_csv('collaborative.csv',  decimal=",")

# %%
import pandas as pd
def openSheetsXlxs(name):
    file = pd.ExcelFile(name)
    listSheets = file.sheet_names
    for elem in listSheets:
         globals()[elem] = pd.read_excel(name, sheet_name=elem)
         globals()[elem].drop(['Unnamed: 0'], axis=1, inplace=True)

    return globals()[elem]

openSheetsXlxs("traffic_rules.xlsx")

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
model = LinearRegression()


#Ordinal
train, test = train_test_split(rules_ordinal, test_size=0.2)
X = train.drop([train.columns[-1]], axis=1)
Y = train.drop(train.columns[0:-1], axis=1) #predict pass

X_test = train.drop([test.columns[-1]], axis=1)
Y_test = train.drop(test.columns[0:-1], axis=1) #predict pass
model.fit(X,Y)
prediction = model.predict(X_test)
print("Ordinary mse " , mean_squared_error(Y_test,prediction))

#Categorical
train, test = train_test_split(rules_categorical, test_size=0.2)
X = train.drop([train.columns[-1]], axis=1)
Y = train.drop(train.columns[0:-1], axis=1) #predict pass

X_test = train.drop([test.columns[-1]], axis=1)
Y_test = train.drop(test.columns[0:-1], axis=1) #predict pass
model.fit(X,Y)
prediction = model.predict(X_test)
print("Categorical mse " , mean_squared_error(Y_test,prediction))
