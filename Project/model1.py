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
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

data = pd.read_csv('model_data.csv')
#Data med T er transformed


In this exercise I want to predict 


##Regression Tree regressor

data.describe()

X_get_away = data[['Squad', 'ScoreAway', 'Age',
       'Goals90min', 'NonPenaltyGoals', 'PenaltyKicksMade', 'YellowCards',
       'RedCards', 'RankingPlace', 'GoalFor', 'GoalAgainst', 'GoalDifference',
       'Pts']]

X_get_home = data[['Squad', 'ScoreAway', 'Age',
       'Goals90min', 'NonPenaltyGoals', 'PenaltyKicksMade', 'YellowCards',
       'RedCards', 'RankingPlace', 'GoalFor', 'GoalAgainst', 'GoalDifference',
       'Pts']]

X_away = data[['ScoreHome', 'Age',
       'Goals90min', 'NonPenaltyGoals', 'PenaltyKicksMade', 'YellowCards',
       'RedCards', 'RankingPlace', 'GoalFor', 'GoalAgainst', 'GoalDifference',
       'Pts']]

X_home = data[['ScoreAway', 'Age',
       'Goals90min', 'NonPenaltyGoals', 'PenaltyKicksMade', 'YellowCards',
       'RedCards', 'RankingPlace', 'GoalFor', 'GoalAgainst', 'GoalDifference',
       'Pts']]


y_away = data['ScoreAway']
y_home = data['ScoreHome']




#AWAY
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_away, y_away, test_size=0.4,  random_state=20)
rt = DecisionTreeRegressor(criterion = 'mse', max_depth=5)
model_r_away = rt.fit(X_train_a, y_train_a)
y_pred_away = model_r_away.predict(X_test_a)



#HOME
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_home, y_home, test_size=0.4,  random_state=20)
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

get_h = X_get_home.loc[(X_get_home['Squad'] == team_home)]
get_h = get_h.drop(columns=['Squad'])
y_pred_home_single = model_r_home.predict(get_h)
yh = (round(y_pred_home_single[0]))

get_a = X_get_away.loc[(X_get_away['Squad'] == team_away)]
get_a = get_a.drop(columns=['Squad'])
y_pred_away_single = model_r_away.predict(get_a)
ya = (round(y_pred_away_single[0]))

print('\n')
print('Home:', team_home, '\n', 'Away:', team_away, '\n', 'RESULT: ', str(yh) ,' : ', str(ya))



model_lasso_a =  linear_model.Lasso(alpha=1.0)
model_lasso_a.fit(X_train_a, y_train_a)
predict_home_away_l = model_lasso_a.predict(X_test_a)


model_lasso_h =  linear_model.Lasso(alpha=1.0)
model_lasso_h.fit(X_train_h, y_train_h)
predict_home_home_l= model_lasso_h.predict(X_test_h)



print('\n')
print('Lasso')
team_away = input('Write team away: ')    
team_home = input('Write team home: ') 


get_h = X_get_home.loc[(X_get_home['Squad'] == team_home)]
get_h = get_h.drop(columns=['Squad'])
y_pred_home_single = model_lasso_h.predict(get_h)
yh = (round(y_pred_home_single[0]))

get_a = X_get_away.loc[(X_get_away['Squad'] == team_away)]
get_a = get_a.drop(columns=['Squad'])
y_pred_away_single = model_lasso_a.predict(get_a)
ya = (round(y_pred_away_single[0]))



print('Home:', team_home, '\n', 'Away:', team_away, '\n', 'RESULT: ', str(yh) ,' : ', str(ya))

# %% MULTI


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
