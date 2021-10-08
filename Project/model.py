#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 16:47:28 2021

@author: edyta
"""

# linear regression for multioutput regression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler



data = pd.read_csv('data.csv')
# Simply drop teams that were not in specific season
data = data.dropna()


X = data[['NonPenaltyGoalsHome', 'AgeHome', 'Goals90minHome',
          'NonPenaltyGoalsAway', 'AgeAway', 'Goals90minAway', 'RankingPlaceHome',
          'GoalForHome', 'GoalAgainstHome', 'GoalDifferenceHome', 'PointsHome',
          'TopTeamScorerGoalsHome', 'RankingPlaceAway', 'GoalForAway',
          'GoalAgainstAway', 'GoalDifferenceAway', 'PointsAway',
          'TopTeamScorerGoalsAway', 'HomeT', 'AwayT', 'Season']]


normaliza = MinMaxScaler() 
X_normal = normaliza.fit_transform(X)
X = pd.DataFrame(X_normal)


y = data[['ScoreHome', 'ScoreAway']]

X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.6)

test_size = 0.5
X_valid, X_test, y_valid, y_test = train_test_split(
    X_rem, y_rem, test_size=0.5, random_state=20)


model = LinearRegression()
model.fit(X_train, y_train)

model_KN = KNeighborsRegressor()
model_KN.fit(X_train, y_train)

model_T = DecisionTreeRegressor()
model_T.fit(X_train, y_train)



# Instantiate model with 1000 decision trees
model_RF = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
model_RF.fit(X_train, y_train);




test = pd.read_csv('2020/test.csv')
test2 = pd.DataFrame(columns=[['Home', 'Away']])

home = []
away = []
for ind in test.index:
    if (test['Venue'][ind]) == 'Home':
        home.append(test['Team'][ind])
        away.append(test['Opponent'][ind])
    else:
        home.append(test['Opponent'][ind])
        away.append(test['Team'][ind])

test['Home'] = home
test['Away'] = away
test = test.drop(columns=['Date', 'Team', 'Opponent', 'Venue'])

y_pred_Linear_csv_home = []
y_pred_Linear_csv_away = []

y_pred_KN_csv_home = []
y_pred_KN_csv_away = []

y_pred_T_csv_home = []
y_pred_T_csv_away = []

y_pred_RF_csv_home = []
y_pred_RF_csv_away = []

team_transformed = pd.DataFrame()
team_transformed['Home'] = data.Home
team_transformed['SquadNO'] = data.HomeT
description = team_transformed.drop_duplicates(
    ['Home', 'SquadNO'], keep='last')


# get X values that will be used in model (need it to predit)
get_data = data[['Home', 'Away', 'NonPenaltyGoalsHome', 'AgeHome', 'Goals90minHome',
                 'NonPenaltyGoalsAway', 'AgeAway', 'Goals90minAway', 'RankingPlaceHome',
                 'GoalForHome', 'GoalAgainstHome', 'GoalDifferenceHome', 'PointsHome',
                 'TopTeamScorerGoalsHome', 'RankingPlaceAway', 'GoalForAway',
                 'GoalAgainstAway', 'GoalDifferenceAway', 'PointsAway',
                 'TopTeamScorerGoalsAway', 'HomeT', 'AwayT', 'Season']]

for ind in test.index:
    get_h = (test['Home'][ind])
    get_a = (test['Away'][ind])
    get = get_data[(get_data['Home'] == get_h) & (get_data['Away'] == get_a)]
    get = get.drop(columns=['Home', 'Away'])
    pred1 = (model.predict(get))
    pred2 = (model_KN.predict(get))
    pred3 = (model_T.predict(get))
    pred4 = (model_RF.predict(get))

    
    
    y_pred_Linear_csv_home.append(round(pred1[0][0]))    
    y_pred_Linear_csv_away.append(round(pred1[0][1]))   
    
    y_pred_KN_csv_home.append(round(pred2[0][0]))   
    y_pred_KN_csv_away.append(round(pred2[0][1]))

    y_pred_T_csv_home.append(round(pred3[0][0]))    
    y_pred_T_csv_away.append(round(pred3[0][1]))    
    
    y_pred_RF_csv_home.append(round(pred4[0][0]))    
    y_pred_RF_csv_away.append(round(pred4[0][1]))




#Create csv
   
test['HomeScore_tree'] = y_pred_T_csv_home
test['AwayScore_tree'] = y_pred_T_csv_away

test['HomeScore_Linear'] = y_pred_Linear_csv_home
test['AwayScore_Linear'] = y_pred_Linear_csv_away

test['HomeScore_KN'] = y_pred_KN_csv_home
test['AwayScore_KN'] = y_pred_KN_csv_away

test['HomeScore_RF'] = y_pred_RF_csv_home
test['AwayScore_RF'] = y_pred_RF_csv_away
test.to_csv(r'/home/edyta/git/INF161/Project/results_data.csv')




# compute the Mean Square Error on both datasets.
y_pred_test = model.predict(X_test)
y_pred_valid = model.predict(X_valid)

test_MSE1 = metrics.mean_squared_error(y_test, y_pred_test, squared=False)
valid_MSE1 = metrics.mean_squared_error(y_valid, y_pred_valid, squared=False)

y_pred_test2 = model_KN.predict(X_test)
y_pred_valid2 = model_KN.predict(X_valid)

test_MSE2 = metrics.mean_squared_error(y_test, y_pred_test2, squared=False)
valid_MSE2 = metrics.mean_squared_error(y_valid, y_pred_valid2, squared=False)

y_pred_test3 = model_T.predict(X_test)
y_pred_valid3 = model_T.predict(X_valid)

test_MSE3 = metrics.mean_squared_error(y_test, y_pred_test3, squared=False)
valid_MSE3 = metrics.mean_squared_error(y_valid, y_pred_valid3, squared=False)

y_pred_test4 = model_RF.predict(X_test)
y_pred_valid4 = model_RF.predict(X_valid)

test_MSE4 = metrics.mean_squared_error(y_test, y_pred_test4, squared=False)
valid_MSE4 = metrics.mean_squared_error(y_valid, y_pred_valid4, squared=False)

error = pd.DataFrame(columns = ['Linear model', 'KNeighborsRegressor', 'DecisionTreeRegressor', 'Random Forest'])
error['Linear model'] = [test_MSE1, valid_MSE1]
error['KNeighborsRegressor'] = [test_MSE2, valid_MSE2]
error['DecisionTreeRegressor'] = [test_MSE3, valid_MSE3]
error['Random Forest'] = [test_MSE4, valid_MSE4]

error.rename(index={0:'Test', 1:'Validation'},inplace=True)

