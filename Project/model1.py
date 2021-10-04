#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 15:13:58 2021

@author: edyta
"""
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor

from sklearn.datasets import make_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

data = pd.read_csv('model_data.csv')
#Data med T er transformed


##Regression Tree regressor

data.describe()

X_get = data[['Squad', 'ScoreHome', 'ScoreAway', 'Age',
       'Goals90min', 'NonPenaltyGoals', 'PenaltyKicksMade', 'YellowCards',
       'RedCards', 'RankingPlace', 'GoalFor', 'GoalAgainst', 'GoalDifference',
       'Pts']]

X = data[['ScoreHome', 'ScoreAway', 'Age',
       'Goals90min', 'NonPenaltyGoals', 'PenaltyKicksMade', 'YellowCards',
       'RedCards', 'RankingPlace', 'GoalFor', 'GoalAgainst', 'GoalDifference',
       'Pts']]


y_away = data['ScoreAway']
y_home = data['ScoreHome']





X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X, y_away, test_size=0.4)
rt = DecisionTreeRegressor(criterion = 'mse', max_depth=5)
model_r_away = rt.fit(X_train_a, y_train_a)
y_pred_away = model_r_away.predict(X_test_a)



X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X, y_home, test_size=0.4)
rt = DecisionTreeRegressor(criterion = 'mse', max_depth=5)
model_r_home = rt.fit(X_train_h, y_train_h)
y_pred_home = model_r_home.predict(X_test_h)


print('Mean Absolute Error away:', metrics.mean_absolute_error(y_test_a, y_pred_away))
print('Mean Absolute Error home:', metrics.mean_absolute_error(y_test_h, y_pred_home))

team_away = input('Write team away: ')    
team_home = input('Write team home: ') 


team_transformed = pd.DataFrame()
team_transformed['Squad'] = data.Squad
team_transformed['SquadNO'] = data.SquadT
description = team_transformed.drop_duplicates(['Squad','SquadNO'], keep='last')

get_h = X_get.loc[(X_get['Squad'] == team_home)]
get_h = get_h.drop(columns=['Squad'])
y_pred_home_single = model_r_home.predict(get_h)
yh = (round(y_pred_home_single[0]))

get_a = X_get.loc[(X_get['Squad'] == team_away)]
get_a = get_a.drop(columns=['Squad'])
y_pred_away_single = model_r_away.predict(get_a)
ya = (round(y_pred_away_single[0]))

print('\n')
print('Home:', team_home, '\n', 'Away:', team_away, '\n', 'RESULT: ', str(yh) ,' : ', str(ya))


