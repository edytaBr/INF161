#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 17:07:32 2021

@author: edyta
"""

import pandas as pd
import statistics

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
    
    

    