#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 15:54:00 2021

@author: edyta
"""

# linear regression for multioutput regression
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split


data = pd.read_csv('model_data.csv')

X = data[['Age',
       'Goals90min', 'NonPenaltyGoals', 'PenaltyKicksMade', 'YellowCards',
       'RedCards', 'RankingPlace', 'GoalFor', 'GoalAgainst', 'GoalDifference',
       'Pts']]


y= data[['ScoreHome', 'ScoreAway']]
#AWAY
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,  random_state=20)

model = LinearRegression()
model.fit(X_train, y_train)


from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor
# create datasets
# define model
model_KN = KNeighborsRegressor()
# fit model
model_KN.fit(X_train, y_train)




# decision tree for multioutput regression
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
# create datasets
# define model
model_T = DecisionTreeRegressor()
# fit model
model_T.fit(X_train, y_train)




team_away = input('Write team away: ')    
team_home = input('Write team home: ') 


team_transformed = pd.DataFrame()
team_transformed['Squad'] = data.Squad
team_transformed['SquadNO'] = data.SquadT
description = team_transformed.drop_duplicates(['Squad','SquadNO'], keep='last')



get_data = data[['Squad', 'Age',
       'Goals90min', 'NonPenaltyGoals', 'PenaltyKicksMade', 'YellowCards',
       'RedCards', 'RankingPlace', 'GoalFor', 'GoalAgainst', 'GoalDifference',
       'Pts']]

get= get_data.loc[(get_data['Squad'] == team_home)]
get =get.drop(columns=['Squad'])
pred_team1 = model.predict(get)

get= get_data.loc[(get_data['Squad'] == team_away)]
get =get.drop(columns=['Squad'])
pred_team2 = model.predict(get)
print('Linear')
print('Home:', team_home, '\n', 'Away:', team_away, '\n', 'RESULT: ', str(pred_team1[0][0]) ,' : ', str(pred_team2[0][1]))
get= get_data.loc[(get_data['Squad'] == team_home)]
get =get.drop(columns=['Squad'])
pred_team1 = model_KN.predict(get)

get= get_data.loc[(get_data['Squad'] == team_away)]
get =get.drop(columns=['Squad'])
pred_team2 = model_KN.predict(get)
print('KN')
print('Home:', team_home, '\n', 'Away:', team_away, '\n', 'RESULT: ', str(pred_team1[0][0]) ,' : ', str(pred_team2[0][1]))
get= get_data.loc[(get_data['Squad'] == team_home)]
get =get.drop(columns=['Squad'])
pred_team1 = model_T.predict(get)
print('Tree')
get= get_data.loc[(get_data['Squad'] == team_away)]
get =get.drop(columns=['Squad'])
pred_team2 = model_T.predict(get)
print('Home:', team_home, '\n', 'Away:', team_away, '\n', 'RESULT: ', str(pred_team1[0][0]) ,' : ', str(pred_team2[0][1]))