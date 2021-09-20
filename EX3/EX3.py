#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 17:34:23 2021

@author: edyta
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


#Read dara from csv
data = pd.read_csv('salary-Industry.csv')
#Reshape data
cs, oil = [x for _, x in data.groupby(data['Industry'])]


cs_x = np.array(cs.YearsExperience).reshape((-1, 1))
cs_y = np.array(cs.Salary).reshape((-1, 1))
plt.scatter(cs_x,cs_y,color='black')
reg_cs=linear_model.LinearRegression()
reg_cs.fit(cs_x,cs_y)
cs_salary_pred = reg_cs.predict(cs_x)



oil_x = np.array(oil.YearsExperience).reshape((-1, 1))
oil_y = np.array(oil.Salary).reshape((-1, 1))
plt.scatter(oil_x,oil_y,color='black')
reg_oil=linear_model.LinearRegression()
reg_oil.fit(oil_x,oil_y)
oil_salary_pred = reg_oil.predict(oil_x)



# Plot outputs
plt.scatter(oil_x, oil_y,  color='red')
plt.plot(oil_x, oil_salary_pred, color='blue', linewidth=3)
plt.scatter(cs_x, cs_y,  color='black')
plt.plot(cs_x, cs_salary_pred, color='red', linewidth=3)
plt.xticks(())
plt.yticks(())






plt.show()