#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 17:03:10 2021

@author: edyta
"""

from sklearn.metrics import mean_squared_error as MSE, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import seaborn as sns
from matplotlib.colors import ListedColormap


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
    X_rem, y_rem, test_size=0.5, random_state=20)

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

plt.style.use('ggplot')
df = pd.DataFrame()
degrees = ([2, 5, 10, 20, 25])

summary = pd.DataFrame()
summary['index name'] = ["Validation", "Train"]
summary = pd.DataFrame(summary.set_index('index name'))

# Visualizing the Polymonial Regression results


X_train = np.array(X_train).reshape(-1, 1)
X_valid = np.array(X_valid).reshape(-1, 1)


def viz_polymonial(deg):
    poly_X_train = PolynomialFeatures(deg).fit_transform(X_train)
    poly_X_valid = PolynomialFeatures(deg).fit_transform(X_valid)
    lr_poly = LinearRegression().fit(poly_X_train, y_train)
    y_train_poly_ = lr_poly.predict(poly_X_train)
    y_valid_poly_ = lr_poly.predict(poly_X_valid)
    plt.figure(figsize=(15, 10))

    sns.scatterplot(X_valid[:, 0], y_valid[:, 0], color='blue', edgecolors='blue',
                    marker="X", label="Validation Data Points")
    sns.lineplot(X_valid[:, 0], y_valid_poly_[:, 0],  color='orange')

    # ERROR
    summary[str(deg) + " MSE"] = [metrics.mean_squared_error(y_valid, y_valid_poly_),
                                  metrics.mean_squared_error(y_train, y_train_poly_)]
    plt.title('Polynominal Regression of degree ' + str(deg))
    plt.xlabel('X')
    plt.ylabel('Y ')
    plt.legend(bbox_to_anchor=(1, 0.8, 0.3, 0.2),
               loc='upper left', facecolor='lavender')
    plt.show()
    return


degrees = ([2, 5, 10, 20, 25])

for i in range(0, len(degrees)):
    viz_polymonial(degrees[i])

summary['Linear'] = [valid_MSE, test_MSE]


# %%


# Visualization:
cmap_light = ListedColormap(['#EB98FD', '#66C5E3', '#F3FB95'])
cmap_dark = ListedColormap(['#8E44AD', '#1B85C5', '#F1EE32'])
k = [1, 5, 10, 20, 30]


iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target
h = 0.02
summary = pd.DataFrame()
summary['index name'] = ["Validation", "Train", "K"]
summary = pd.DataFrame(summary.set_index('index name'))

# Split data
X_train, X_rem, y_train, y_rem = train_test_split(X, Y, train_size=0.6)

test_size = 0.5
X_valid, X_test, y_valid, y_test = train_test_split(
    X_rem, y_rem, test_size=0.5, random_state=20)


def viz_classification(k):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    valid_score = accuracy_score(y_valid, model.predict(X_valid))
    train_score = accuracy_score(y_train, model.predict(X_train))
    summary[str(k) + " Accurancy Score"] = [valid_score, train_score, k]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_dark, edgecolors='black')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title('-NN classification of your dataset for k = ' + str(k))
    plt.xlabel('sepal length (cm)')
    plt.ylabel('sepal width (cm)')
    plt.legend(bbox_to_anchor=(1, 0.8, 0.3, 0.2),
               loc='upper left', facecolor='lavender')
    return


for i in range(0, len(k)):
    viz_classification(k[i])

plt.figure()
plt.subplots_adjust(left=0.1)
plt.plot(summary.iloc[2], summary.iloc[1], label="Validation")
plt.plot(summary.iloc[2], summary.iloc[0], label="Train")
plt.title("Change in as a function of different K")

plt.legend(loc='best', facecolor='lavender')
plt.show()
