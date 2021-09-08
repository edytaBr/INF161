#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 17:07:32 2021

@author: edyta
"""

import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv ('Real estate.csv')
#print (df)

df.describe()

#Not that not all columns give meaning to summary

#summary table

summary = []

columns = df.columns.values.tolist()

column_names = ["Column", "Mean", "Median", "Quantile 75", "Variance", "SD"]
summary = pd.DataFrame(columns = column_names)
summary['Column'] = columns

for x in range(df.shape[1]):
    data = (df.iloc[:, [x]])
    summary['Mean'][x] = (data.mean()[0])
    summary['Median'][x] = (data.median()[0])
    summary['Quantile 75'][x]= (data.quantile(.75)[0])
    summary['Variance'][x] = (data.var()[0])
    summary['SD'][x] = (data.std()[0])
    
##2
df.isnull().values.any()
X = df.drop('Y house price of unit area',axis=1)
y = df['Y house price of unit area']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=101)

model= LinearRegression()
model.fit(X_train, y_train)
LinearRegression()
pd.DataFrame(model.coef_, X.columns, columns=['Coeficient'])
y_pred=model.predict(X_test)


houseToPredict = pd.DataFrame()
prediction = {'No': 1, 'X1 transaction date': 2014, 'X2 house age': 10, 
              'X3 distance to the nearest MRRT station' : 1200, 'X4 number of convenience stores': 5, 
              'X5 latitude': 24.93, 
              'X6 longitude': 121.54}

prediction = {'No': 1, 'X1 transaction date': 2014, 'X2 house age': 10, 
              'X3 distance to the nearest MRRT station' : 1200, 'X4 number of convenience stores': 5, 
              'X5 latitude': 24.93, 
              'X6 longitude': 121.54}




houseToPredict = houseToPredict.append(prediction, ignore_index = True)
predictedPrice=model.predict(houseToPredict)


