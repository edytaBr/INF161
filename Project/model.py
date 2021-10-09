#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 16:47:28 2021

@author: edyta
"""

# linear regression for multioutput regression
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np

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
    X_rem, y_rem, test_size=0.5)


model = LinearRegression()
model_KN = KNeighborsRegressor()
model_T = DecisionTreeRegressor()
model_RF = RandomForestRegressor(n_estimators = 1000, random_state = 42)
en = ElasticNet()




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






def run_model_calc_errors(model,X_train,y_train,X_val,y_val,X_test,y_test):

    model.fit(X=X_train, y=y_train)

    y_pred_train=model.predict(X_train)
    MSE_train=mean_squared_error(y_train, y_pred_train)

    y_pred_val=model.predict(X_val)
    MSE_val=mean_squared_error(y_val, y_pred_val)
  

    y_pred_test=model.predict(X_test)
    MSE_test=mean_squared_error(y_test, y_pred_test)
    RMSE=np.sqrt(MSE_test)
    
    return [MSE_train, MSE_val, RMSE]

models = [model, model_KN, model_RF, model_T,en]


error = pd.DataFrame(columns=['MSE_Train', 'MSE_Valid', 'RMSE_Test'])
for model in models:
    res = run_model_calc_errors(model, X_train, y_train, X_valid, y_valid, X_test, y_test)
    error = error.append({'MSE_Train': res[0],'MSE_Valid': res[1], 'RMSE_Test': res[2]}, ignore_index=True) 





error.rename(index={0:'LinearRegression',
                    1:'KNeighborsRegressor',
                    2: 'RandomForestRegressor',
                    3: 'DecisionTreeRegressor',
                    4: 'ElasticNet' },inplace=True)






#CSV


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

test['HomeScore_tree'] = y_pred_T_csv_home
test['AwayScore_tree'] = y_pred_T_csv_away

test['HomeScore_Linear'] = y_pred_Linear_csv_home
test['AwayScore_Linear'] = y_pred_Linear_csv_away

test['HomeScore_KN'] = y_pred_KN_csv_home
test['AwayScore_KN'] = y_pred_KN_csv_away

test['HomeScore_RF'] = y_pred_RF_csv_home
test['AwayScore_RF'] = y_pred_RF_csv_away
test.to_csv(r'/home/edyta/git/INF161/Project/results_data.csv')





