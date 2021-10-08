#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 10:38:40 2021

@author: edyta
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn import preprocessing



def prepareGameData(season):
    if season == 2020:
        games = prepare2020GameData(2020)
        return games
    else:
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

def prepare2020GameData(season):
    games = pd.read_html(str(season) + '/games.xls', encoding='utf-8')[0]
    games['Season']= str(season)
    games.dropna(inplace=True)

    games = games.drop(columns=['Wk',
                                'Time',
                                'Date',
                                'Venue'])
    return games    

        
games17 = prepareGameData(2017)
games18 = prepareGameData(2018)
games19 = prepareGameData(2019)

games17 = prepareGameData(2017)
games18 = prepareGameData(2018)
games19 = prepareGameData(2019)
games20 = prepareGameData(2020)

g20 = games19.merge(games20[['Home', 'Away']])
g20.Season = str(2020)
games20 = g20




def joinDataFrames(df1, df2, df3, df4):
    df = df1.append(df2, ignore_index = True)
    df = df.append(df3, ignore_index = True)
    df = df.append(df4, ignore_index = True)

    return df

data = joinDataFrames(games17, games18, games19, games20)
games = data.rename(columns={'Squad': 'Home'}) #change name of Home to Squad

data_grouped  = data.rename(columns={'Home': 'Squad'}) #change name of Home to Squad
games1 = data_grouped.groupby(['Season', 'Squad']).mean() #group to generalize and match with other dfs.

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

teamStats20 = teamStats19.copy()
teamStats20.Season = str(2020)

teamStats = joinDataFrames(teamStats17, teamStats18, teamStats19, teamStats20)

data_vis = games1.merge(teamStats,  on=['Season', 'Squad'])



teamStats = joinDataFrames(teamStats17, teamStats18, teamStats19, teamStats20)


teamStats  = teamStats.rename(columns={'PK': 'PenaltyKicksMade', 
                                         'G-PK': 'NonPenaltyGoals', 
                                         'Gls': 'Goals90min',
                                         'CrdY': 'YellowCards',
                                         'CrdR': 'RedCards'}) 



teamStats = teamStats[['NonPenaltyGoals', 'Age', 'Goals90min', 'Squad', 'Season']]
#HOME CASE
teamStats  = teamStats.rename(columns={'Squad': 'Home'}) 
merged = pd.merge(games, teamStats, how="outer", on=["Home", "Season"])
merged.head()
merged.shape

merged  = merged.rename(columns={'NonPenaltyGoals': 'NonPenaltyGoalsHome', 
                           'Age': 'AgeHome',
                           'Goals90min': 'Goals90minHome'}) 

#AWAY case
teamStats  = teamStats.rename(columns={'Home': 'Away'}) 
merged = pd.merge(merged, teamStats, how="outer", on=["Away", "Season"])

merged  = merged.rename(columns={'NonPenaltyGoals': 'NonPenaltyGoalsAway', 
                           'Age': 'AgeAway',
                           'Goals90min': 'Goals90minAway'}) 






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
                    'Top Team Scorer Goals']]
     table['Top Team Scorer Goals']= table['Top Team Scorer Goals'].astype(int)
     table['Season']= str(season)
     return table

table17 = prepareTableData(2017)
table18 = prepareTableData(2018)
table19 = prepareTableData(2019)
table20 = table19.copy()
table20.Season = str(2020)

table = joinDataFrames(table17, table18, table19, table20)

table  = table.rename(columns={'GF': 'GoalFor', 
                               'GA': 'GoalAgainst', 
                               'GD': 'GoalDifference', 
                               'Rk': 'RankingPlace',
                               'Top Team Scorer Goals': 'TopTeamScorerGoals', 
                                'Pts': 'Points'}) 



data_vis = data_vis.merge(table,  on=['Season', 'Squad'])
data_vis  = data_vis.rename(columns={'PK': 'PenaltyKicksMade', 
                                         'G-PK': 'NonPenaltyGoals', 
                                         'GF': 'GoalFor', 
                                         'GA': 'GoalAgainst', 
                                         'GD': 'GoalDifference', 
                                         'Rk': 'RankingPlace',
                                         'Gls': 'Goals90min',
                                         'CrdY': 'YellowCards',
                                         'CrdR': 'RedCards'}) 
#HOME CASE
table  = table.rename(columns={'Squad': 'Home'}) 
merged = pd.merge(merged, table, how="outer", on=["Home", "Season"])
merged.head()
merged.shape

merged  = merged.rename(columns={'GoalFor': 'GoalForHome', 
                           'GoalAgainst': 'GoalAgainstHome',
                           'GoalDifference': 'GoalDifferenceHome',
                           'RankingPlace': 'RankingPlaceHome',
                           'TopTeamScorerGoals': 'TopTeamScorerGoalsHome',
                            'Points': 'PointsHome'}) 

#AWAY case
table  = table.rename(columns={'Home': 'Away'}) 
merged = pd.merge(merged, table, how="outer", on=["Away", "Season"])
merged.head()
merged.shape

data  = merged.rename(columns={'GoalFor': 'GoalForAway', 
                           'GoalAgainst': 'GoalAgainstAway',
                           'GoalDifference': 'GoalDifferenceAway',
                           'RankingPlace': 'RankingPlaceAway',
                           'TopTeamScorerGoals': 'TopTeamScorerGoalsAway',
                            'Points': 'PointsAway'}) 

number = preprocessing.LabelEncoder()
data['HomeT'] = number.fit_transform(data.Home)
data['HomeT'] = number.fit_transform(data.Home)
data['AwayT'] = number.fit_transform(data.Away)
data['AwayT'] = number.fit_transform(data.Away)

data.to_csv(r'/home/edyta/git/INF161/Project/data.csv')

data_vis = data_vis.reset_index()

plots = data_vis[['Squad', 'Season', 'ScoreHome', 'ScoreAway', 'Age', 'Goals90min',
       'NonPenaltyGoals', 'PenaltyKicksMade', 'YellowCards', 'RedCards',
       'RankingPlace', 'GoalFor', 'GoalAgainst', 'GoalDifference', 'Points',
       'TopTeamScorerGoals']]


v = plots.reset_index()
v = v.sort_values(by='ScoreHome', ascending=False)


fig_1 = go.Figure(data=[
    go.Bar(name='2017',x=v.Squad[v.Season == '2017'], y = v.ScoreAway[v.Season == '2017']),
    go.Bar(name='2018',x=v.Squad[v.Season == '2018'], y = v.ScoreAway[v.Season == '2018']),
    go.Bar(name='2019',x=v.Squad[v.Season == '2019'], y = v.ScoreAway[v.Season == '2019']),
   ])

fig_1.update_layout(title_text='Goals away for teams 2017-2019')
fig_1.show()

v = v.sort_values(by='ScoreAway', ascending=False)

fig_1a = go.Figure(data=[
    go.Bar(name='2017',x=v.Squad[v.Season == '2017'], y = v.ScoreHome[v.Season == '2017']),
    go.Bar(name='2018',x=v.Squad[v.Season == '2018'], y = v.ScoreHome[v.Season == '2018']),
    go.Bar(name='2019',x=v.Squad[v.Season == '2019'], y = v.ScoreHome[v.Season == '2019']),
   ])

fig_1a.update_layout(title_text='Goals home for teams 2017-2019')
fig_1a.show()

fig_2 = px.scatter(
        v, x="Squad", y="GoalFor", 
        size =list(map(int, v['TopTeamScorerGoals'])),  color=list(map(int, v['TopTeamScorerGoals'])), 
        hover_data=['Season'])

fig_2.update_layout(title_text='Goals for teams in 2017-2019 with goals scored by the best scorer in the team (color)')
fig_2.show()

correlation = v.iloc[:, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]].corr()
z=correlation[['ScoreHome', 'ScoreAway']]
fig_3 = go.Figure(data=go.Heatmap(z=correlation[['ScoreHome', 'ScoreAway']], x=z.columns, y =correlation.columns,  colorscale='RdBu_r'))
fig_3.update_layout(title_text='Correlation plot for particular statistic for teams 2017 example')
fig_3.show()