#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 17:17:54 2021

@author: edyta
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.renderers.default='browser'

#Scores to get it as integer in separate columns
#GAMES: LOOP

def prepareGameData(season):
    games = pd.read_html(str(season) + '/games.xls', encoding='utf-8')[0]
    games.dropna(subset = ["Score"], inplace=True)
    games[['ScoreHome', 'ScoreAway']] = games['Score'].str.split('–', 1, expand=True)
    games[['ScoreHome', 'ScoreAway']]= games[['ScoreHome', 'ScoreAway']].astype(int)
    games['Season']= str(season)
    if set(['Round']).issubset(games.columns):
        games = games.drop(columns=['Wk', 
                                    'Notes', 
                                    'Match Report', 
                                    'Score',
                                    'Round',
                                    'Referee', 
                                    'Venue',
                                    'Date', 
                                    'Day',
                                    'Time',
                                    'Attendance'])
    else:
        games = games.drop(columns=['Wk', 
                                    'Notes',
                                    'Match Report',
                                    'Score',
                                    'Referee',
                                    'Venue',
                                    'Date',
                                    'Day',
                                    'Time',
                                    'Attendance'])
    return games    
    
        
games17 = prepareGameData(2017)
games18 = prepareGameData(2018)
games19 = prepareGameData(2019)


def joinDataFrames(df1, df2, df3):
    df = df1.append(df2, ignore_index = True)
    df = df.append(df3, ignore_index = True)
    return df

data = joinDataFrames(games17, games18, games19)
data_grouped  = data.rename(columns={'Home': 'Squad'}) #change name of Home to Squad
games = data_grouped.groupby(['Season', 'Squad']).mean() #group to generalize and match with other dfs.



def prepareTeamStatData(season):
     teamStats = pd.read_html(str(season) + '/team-stats.xls', encoding='utf-8',header=1)[0]
     teamStats.dropna(axis=1 , how='all').dropna(axis=0 , how='all')   
     teamStats['Season']= str(season)
     teamStats = teamStats[['Squad',
                            'Age',
                            'Gls',
                            'G-PK',
                            'PK',
                            'CrdY',
                            'CrdR',
                            'Season']]
     return teamStats

teamStats17 = prepareTeamStatData(2017)
teamStats18 = prepareTeamStatData(2018)
teamStats19 = prepareTeamStatData(2019)
teamStats = joinDataFrames(teamStats17, teamStats18, teamStats19)

data_model = games.merge(teamStats,  on=['Season', 'Squad'])



def prepareTableData(season):
     table = pd.read_html(str(season) + '/table.xls', encoding='utf-8')[0]
     table.dropna(axis=1 , how='all').dropna(axis=0 , how='all')   
     table[['Top Team Scorer', 'Top Team Scorer Goals']] = table['Top Team Scorer'].str.split(' - ', 1, expand=True)
     table = table[['Rk', 
                    'Squad',
                    'GF', 
                    'GA',
                    'GD',
                    'Pts',
                    'Top Team Scorer',
                    'Top Team Scorer Goals',
                    'Goalkeeper']]
     table['Top Team Scorer Goals']= table['Top Team Scorer Goals'].astype(int)
     table['Season']= str(season)
     return table

table17 = prepareTableData(2017)
table18 = prepareTableData(2018)
table19 = prepareTableData(2019)
table = joinDataFrames(table17, table18, table19)

data_model = data_model.merge(table,  on=['Season', 'Squad'])
data_model  = data_model.rename(columns={'PK': 'PenaltyKicksMade', 
                                         'G-PK': 'NonPenaltyGoals', 
                                         'GF': 'GoalFor', 
                                         'GA': 'GoalAgainst', 
                                         'GD': 'GoalDifference', 
                                         'Rk': 'RankingPlace',
                                         'Gls': 'Goals90min',
                                         'CrdY': 'YellowCards',
                                         'CrdR': 'RedCards'}) 

from sklearn import preprocessing




data_model.to_csv(r'/home/edyta/git/INF161/Project/model_data.csv')






#Vizual
vizual = data_model[['Squad', 'Season', 'ScoreHome', 'ScoreAway', 'Age', 'Goals90min',
       'NonPenaltyGoals', 'PenaltyKicksMade', 'YellowCards', 'RedCards',
       'RankingPlace', 'GoalFor', 'GoalAgainst', 'GoalDifference', 'Pts',
       'Top Team Scorer Goals']]


v = vizual.reset_index()


fig_1 = go.Figure(data=[
    go.Bar(name='2017',x=v.Squad[v.Season == '2017'], y = v.ScoreAway[v.Season == '2017']),
    go.Bar(name='2018',x=v.Squad[v.Season == '2018'], y = v.ScoreAway[v.Season == '2018']),
    go.Bar(name='2019',x=v.Squad[v.Season == '2019'], y = v.ScoreAway[v.Season == '2019']),
   ])

fig_1.update_layout(title_text='Goals away for teams 2017-2019')
#fig_1.show()

fig_2 = px.scatter(
        v, x="Squad", y="GoalFor", 
        size =list(map(int, v['Top Team Scorer Goals'])),  color=list(map(int, v['Top Team Scorer Goals'])), 
        hover_data=['Season'])

fig_2.update_layout(title_text='Goals for teams in 2017-2019 with goals scored by the best scorer in the team (color)')
#fig_2.show()

correlation = v.iloc[:, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]].corr()
fig_3 = go.Figure(data=go.Heatmap(z=correlation, x=correlation.columns, y =correlation.columns))
fig_3.update_layout(title_text='Correlation plot for particular statistic for teams 2017 example')
#fig_3.show()

number = preprocessing.LabelEncoder()
data_model['SquadT'] = number.fit_transform(data_model.Squad)
data_model['GoalkeeperT'] = number.fit_transform(data_model.Goalkeeper)
data_model['TopScorrerT'] = number.fit_transform(data_model['Top Team Scorer'])

test = pd.DataFrame()

test[['ScoreHome', 'ScoreAway']] = data_model.groupby(['Squad'])[['ScoreHome', 'ScoreAway']].median()
test[['Age', 'Goals90min',
       'NonPenaltyGoals', 'PenaltyKicksMade', 'YellowCards', 'RedCards',
       'RankingPlace', 'GoalFor', 'GoalAgainst', 'GoalDifference', 'Pts',
       'Top Team Scorer Goals']] = data_model.groupby(['Squad'])[['Age', 'Goals90min',
       'NonPenaltyGoals', 'PenaltyKicksMade', 'YellowCards', 'RedCards',
       'RankingPlace', 'GoalFor', 'GoalAgainst', 'GoalDifference', 'Pts',
       'Top Team Scorer Goals']].mean()
    
test['SquadT'] = data_model.groupby(['Squad'])['SquadT'].median()


data_model = test
data_model.to_csv(r'/home/edyta/git/INF161/Project/model_data.csv')

# Vizualization
data_model = data_model.reset_index()

team_transformed = pd.DataFrame()
team_transformed['Squad'] = data_model.Squad
team_transformed['SquadNO'] = data_model.SquadT
team_description = team_transformed.drop_duplicates(['Squad','SquadNO'], keep='last')
# %% 
