#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 14:07:14 2021

@author: edyta
"""
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier



data = pd.read_csv('diabetes.csv')
list_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
#a)
list_counts = []
for elem in list_columns:
    count = (data[elem] == 0).sum()
    list_counts.append(count)

df = pd.DataFrame(list_counts)
df['index name'] = list_columns
df = pd.DataFrame(df.set_index('index name'))

#b)
data = data[list_columns]
data_nan = data.replace(0, np.NaN)

#c)
from matplotlib import pyplot as plt
import seaborn as sns, numpy as np
sns.set()


def plots(data, data2):
    count=1
    plt.subplots(figsize=(10, 10))
    plt.suptitle('The distribution of columns with and without missing values')
    plt.subplots_adjust(hspace=1, wspace = 0.5)



    for elem in data:
        plt.subplot(3,2,count)
        sns.distplot(data[elem], label='With NaN')
        sns.distplot(data2[elem], label = 'Without NaN')
        count+=1
        plt.show()
    plt.legend(loc='upper center', bbox_to_anchor=(1.5, 1.05))

    
plots(data, data_nan)
#Explain why it is important to use Nan instead of zero for missing values indication.

#By defininf NaNs values we will be sure that they will not be considerd  as normal value, but as NaN and they will not have 
#Infuence on our analysis. 

# Split data
#Split the rows of the dataset into train, validation, and test sets with cor-
#responding ratios of 0.6, 0.2, 0.2, respectively.
# %%
#768 columns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer

data = pd.read_csv('diabetes.csv')

X = data.drop('Outcome', axis=1)
Y = data['Outcome']
# Split data
X_train, X_rem, y_train, y_rem = train_test_split(X, Y, train_size=0.6)
test_size = 0.5
X_valid, X_test, y_valid, y_test = train_test_split(
    X_rem, y_rem, test_size=0.5, random_state=20)

knn = KNeighborsClassifier()


#mean
imp_mean = SimpleImputer(missing_values=np.NaN, strategy='mean')
imp_mean.fit(X_train)

X_train_mean_imp = imp_mean.transform(X_train)
X_val_mean_imp = imp_mean.transform(X_valid)

knn.fit(X_train_mean_imp, y_train)
y_val_mean = knn.predict(X_val_mean_imp)
print(accuracy_score(y_valid, y_val_mean))




#median
imp_median = SimpleImputer(missing_values=np.NaN, strategy='median')
imp_median.fit(X_train)

X_train_median_imp = imp_median.transform(X_train)
X_val_median_imp = imp_median.transform(X_valid)

knn.fit(X_train_median_imp, y_train)
y_val_median = knn.predict(X_val_median_imp)
print(accuracy_score(y_valid, y_val_median))



#most_frequent
imp_mf = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
imp_mf.fit(X_train)

X_train_mf_imp = imp_mf.transform(X_train)
X_val_mf_imp = imp_mf.transform(X_valid)

knn.fit(X_train_mf_imp, y_train)
y_val_mf = knn.predict(X_val_mf_imp)
print(accuracy_score(y_valid, y_val_mf))
