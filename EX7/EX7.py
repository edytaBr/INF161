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

ratings = pd.read_csv('ratings2.csv')
films = pd.read_csv('films.csv')
item = pd.read_csv('item_based.csv',  decimal=",")
coll = ratings - ratings.mean()
coll.drop(['Unnamed: 0'], axis=1, inplace=True)

corr = round(coll.corr(method = 'pearson'), 2)
collaborative = pd.read_csv('collaborative.csv',  decimal=",")
