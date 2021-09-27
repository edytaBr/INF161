#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 17:03:10 2021

@author: edyta
"""

from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# %%
# EXERCISE 1
data = pd.read_csv('regression_nonlin.csv')
plt.style.use('ggplot')

# Let's say we want to split the data in 60:20:20 for train:valid:test dataset
train_size = 0.6


x = np.array(data.X).reshape((-1, 1))
y = np.array(data.y).reshape((-1, 1))

# Validation set is different from test set. Validation set actually can be regarded as a part of training se
# In the first step we will split the data in training and remaining dataset
X_train, X_rem, y_train, y_rem = train_test_split(x, y, train_size=0.6)

# Now since we want the valid and test size to be equal (10% each of overall data).
# we have to define valid_size=0.5 (that is 50% of remaining data)
test_size = 0.5
X_valid, X_test, y_valid, y_test = train_test_split(
    X_rem, y_rem, test_size=0.5, random_state=10)

# Fit the model over the training dataset
model = LinearRegression()
model.fit(X_train, y_train)

fig = plt.figure()

plt.subplots_adjust(wspace=0.2, hspace=0.5)

fig.suptitle('Figure 1: Linear regression on the dataset')

ax1 = fig.add_subplot(211)
ax1.scatter(X_train, y_train,  color='blue', edgecolors='black')
ax1.set_title('Training dataset')
ax1.set_ylabel("Y")
ax1.set_xlabel("X")
ax1.set_xlim(-3, 3)
plt.plot(X_test, model.predict(X_test), color='orange')


ax2 = fig.add_subplot(212)
ax2.scatter(X_valid, y_valid,  color='pink',
            edgecolors='black', label='Oil&Gas')
ax2.set_title('Validation dataset')
ax2.set_ylabel("Y")
ax2.set_xlabel("X")
ax2.set_xlim(-3, 3)
plt.plot(X_valid, model.predict(X_valid), color='orange')

y_pred_test = model.predict(X_test)
y_pred_valid = model.predict(X_valid)

# compute the Mean Square Error on both datasets.

test_MSE = metrics.mean_squared_error(y_test, y_pred_test)
valid_MSE = metrics.mean_squared_error(y_valid, y_pred_valid)

# Polynominal regression
# Fitting Polynomial Regression to the dataset
# %%


# Importing the dataset
# Importing the dataset


X = data.iloc[:, 0].values
y = data.iloc[:, 1].values

X = np.array(X).reshape((-1, 1))


plt.style.use('ggplot')
df = pd.DataFrame()
summary = pd.DataFrame()
summary['index name'] = ["Valid", "Test"]
summary = pd.DataFrame(summary.set_index('index name'))

# Visualizing the Polymonial Regression results


def viz_polymonial(deg):
    fig, ax = plt.subplots(figsize=(8, 8))

    fig.subplots_adjust(top=0.885,
                        bottom=0.11,
                        left=0.11,
                        right=0.815,
                        hspace=0.18,
                        wspace=0.2)
    poly_reg = PolynomialFeatures(degree=deg)
    X_poly = poly_reg.fit_transform(X)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, y)
    df[str(deg) + " Validation"] = pol_reg.predict(poly_reg.fit_transform(X_valid))
    df[str(deg) + " Test"] = pol_reg.predict(poly_reg.fit_transform(X_test))
    aaa= metrics.mean_squared_error(y_test, df[str(deg) + " Validation"])
    summary[str(deg) + " MSE"] = [metrics.mean_squared_error(y_valid, pol_reg.predict(poly_reg.fit_transform(y_valid))), metrics.mean_squared_error(y_test, pol_reg.predict(poly_reg.fit_transform(y_test)))]

    plt.scatter(X_valid, y_valid, color='lightblue', edgecolors='blue',
                marker="X", label="Validation Data Points")  # plot results on validation
    plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X)), color='orange')
    plt.title('Polynominal Regression of degree ' + str(deg))
    plt.xlabel('X')
    plt.ylabel('Y ')
    plt.legend(bbox_to_anchor=(1, 0.8, 0.3, 0.2),
               loc='upper left', facecolor='lavender')
    plt.show()
    return aaa


degrees = ([2, 5, 10, 20, 25])

for i in range(0, len(degrees)):
    viz_polymonial(degrees[i])

summary['Linear'] = [valid_MSE, test_MSE]
# %%
#Compute the Mean Square Error on training and validation datasets for
#each degree and compare it to simple linear model. Does any of your
#models overfit or underfit?
df.(columns=['index name', 'col1',  'col2']).set_index('index name')





