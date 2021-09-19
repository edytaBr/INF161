#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 12:48:50 2021

@author: edyta
"""

import pandas as pd
import dd
import matplotlib.pyplot as plt
import codecs
from pyecharts.charts import Bar
from pyecharts import options as opts
from pyecharts.globals import ThemeType
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns


# %%

games17 = pd.read_html('2017/games.xls', encoding='utf-8')[0]
games18 = pd.read_html('2018/games.xls', encoding='utf-8')[0]
games19 = pd.read_html('2019/games.xls', encoding='utf-8')[0]
#games20 = pd.read_html('2020/games.xls')[0]



#Cleaning games
games17.dropna(subset = ["Score"], inplace=True)
games18.dropna(subset = ["Score"], inplace=True)
games19.dropna(subset = ["Score"], inplace=True)


#Scores to get it as integer in separate columns
games17[['ScoreHome', 'ScoreAway']] = games17['Score'].str.split('–', 1, expand=True)
games17[['ScoreHome', 'ScoreAway']]= games17[['ScoreHome', 'ScoreAway']].astype(int)

games18[['ScoreHome', 'ScoreAway']] = games18['Score'].str.split('–', 1, expand=True)
games18[['ScoreHome', 'ScoreAway']] = games18[['ScoreHome', 'ScoreAway']].astype(int)

games19[['ScoreHome', 'ScoreAway']] = games19['Score'].str.split('–', 1, expand=True)
games19[['ScoreHome', 'ScoreAway']]= games19[['ScoreHome', 'ScoreAway']].astype(int)


#Clean unused columns
games17 = games17.drop(columns=['Notes', 'Match Report', 'Score'])
games18 = games18.drop(columns=['Notes', 'Match Report', 'Score'])
games19 = games19.drop(columns=['Notes', 'Match Report', 'Score'])

games17_grouped = games17.groupby(by=["Home"]).sum()
games18_grouped = games18.groupby(by=["Home"]).sum()
games19_grouped = games19.groupby(by=["Home"]).sum()


games17_grouped_mean = games17.groupby(by=["Home"]).mean()
games18_grouped_mean = games18.groupby(by=["Home"]).mean()
games19_grouped_mean = games19.groupby(by=["Home"]).mean()


games_grouped = pd.DataFrame({
    'Scores Home 17': games17_grouped['ScoreHome'],
    'Scores Away 17': games17_grouped['ScoreAway'],
    'Scores Home 18': games18_grouped['ScoreHome'],
    'Scores Away 18': games18_grouped['ScoreAway'],
    'Scores Home 19': games19_grouped['ScoreHome'],
    'Scores Away 19': games19_grouped['ScoreAway'],
    'Scores mean Home 17': games17_grouped_mean['ScoreHome'],
    'Scores mean Away 17': games17_grouped_mean['ScoreAway'],
    'Scores mean Home 18': games18_grouped_mean['ScoreHome'],
    'Scores mean Away 18': games18_grouped_mean['ScoreAway'],
    'Scores mean Home 19': games19_grouped_mean['ScoreHome'],
    'Scores mean Away 19': games19_grouped_mean['ScoreAway']
})




fig0a = go.Figure(data=[
    go.Bar(name='2017',x=games_grouped.index, y=games_grouped['Scores Away 17']),
    go.Bar(name='2018',x=games_grouped.index, y=games_grouped['Scores Away 18']),
    go.Bar(name='2019',x=games_grouped.index, y=games_grouped['Scores Away 19'])

])

fig0a.update_layout(title_text='Goals away for teams 2017-2019')
fig0a.show()

fig0a = go.Figure(data=[
    go.Bar(name='2017',x=games_grouped.index, y=games_grouped['Scores Home 17']),
    go.Bar(name='2018',x=games_grouped.index, y=games_grouped['Scores Home 18']),
    go.Bar(name='2019',x=games_grouped.index, y=games_grouped['Scores Home 19'])

])

fig0a.update_layout(title_text='Goals home for teams 2017-2019')
fig0a.show()


The games.xls contains the following columns:

- Round: Phase of competition
- Wk: Matchweek number
- Day: Day of the week 
- Date: Match day
- Time: Match time
- Home: Home squad
- Score: Outcome variable
- Away: Away squad
- Attendance: Number of spectators
- Venue: Where the match was played
- Referee: Referee of the match
- Match report: No data available
- Notes: Additional notes

It has been decided to split Score and divide it directly to `Score Home` and `Score Away`. By this the games grouped DataFrame is created.
It is known that some rows are nan since some teams did not qualified to stay in league and on the other hand some teams got opportunity to try in higher division.
Moreover mean scores are calculated for each of team, season and `type` of score: home or away. 
# %% 


#Cleaning stat


playerStats17 = pd.read_html('2017/player-stats.xls', encoding='utf-8',header=1)[0]
playerStats18 = pd.read_html('2018/player-stats.xls', encoding='utf-8', header=1)[0]
playerStats19 = pd.read_html('2019/player-stats.xls', encoding='utf-8', header=1)[0]
playerStats20 = pd.read_html('2020/player-stats.xls', encoding='utf-8', header=1)[0]

playerStats17 = playerStats17.drop(columns=['Matches'])
playerStats18 = playerStats18.drop(columns=['Matches'])
playerStats19 = playerStats19.drop(columns=['Matches'])

#Drop nan
playerStats17 = playerStats17.dropna(axis=1 , how='all').dropna(axis=0 , how='all')
playerStats18 = playerStats18.dropna(axis=1, how='all').dropna(axis=0 , how='all')
playerStats19 = playerStats19.dropna(axis=1, how='all').dropna(axis=0 , how='all')
playerStats20 = playerStats20.dropna(axis=1, how='all').dropna(axis=0 , how='all')



#Unified nation
playerStats17['Nation'] = playerStats17['Nation'].str[-3:]
playerStats18['Nation'] = playerStats18['Nation'].str[-3:]
playerStats19['Nation'] = playerStats19['Nation'].str[-3:]
playerStats20['Nation'] = playerStats20['Nation'].str[-3:]


playerStats17['Pos'].unique()
playerStats17Team = playerStats17.groupby(by=["Squad"]).sum()
playerStats17Team['Defender'] = playerStats17.groupby(['Squad'])['Pos'].apply(lambda x: x[x.str.contains('DF')].count())
playerStats17Team['Midfielders'] = playerStats17.groupby(['Squad'])['Pos'].apply(lambda x: x[x.str.contains('MF')].count())
playerStats17Team['Forwards'] = playerStats17.groupby(['Squad'])['Pos'].apply(lambda x: x[x.str.contains('FW')].count())
playerStats17Team['Goalkeeper'] = playerStats17.groupby(['Squad'])['Pos'].apply(lambda x: x[x.str.contains('FW')].count())
playerStats17Team['Defend & Midfield'] = playerStats17.groupby(['Squad'])['Pos'].apply(lambda x: x[x.str.contains('DF,MF')].count())
playerStats17Team['Forwards & Midfield'] = playerStats17.groupby(['Squad'])['Pos'].apply(lambda x: x[x.str.contains('FW,MF')].count())




playerStats18Team = playerStats18.groupby(by=["Squad"]).sum()
playerStats18Team['Defender'] = playerStats18.groupby(['Squad'])['Pos'].apply(lambda x: x[x.str.contains('DF')].count())
playerStats18Team['Midfielders'] = playerStats18.groupby(['Squad'])['Pos'].apply(lambda x: x[x.str.contains('MF')].count())
playerStats18Team['Forwards'] = playerStats18.groupby(['Squad'])['Pos'].apply(lambda x: x[x.str.contains('FW')].count())
playerStats18Team['Goalkeeper'] = playerStats18.groupby(['Squad'])['Pos'].apply(lambda x: x[x.str.contains('FW')].count())
playerStats18Team['Defend & Midfield'] = playerStats18.groupby(['Squad'])['Pos'].apply(lambda x: x[x.str.contains('DF,MF')].count())
playerStats18Team['Forwards & Midfield'] = playerStats18.groupby(['Squad'])['Pos'].apply(lambda x: x[x.str.contains('FW,MF')].count())



playerStats19Team = playerStats19.groupby(by=["Squad"]).sum()
playerStats19Team['Defender'] = playerStats19.groupby(['Squad'])['Pos'].apply(lambda x: x[x.str.contains('DF')].count())
playerStats19Team['Midfielders'] = playerStats19.groupby(['Squad'])['Pos'].apply(lambda x: x[x.str.contains('MF')].count())
playerStats19Team['Forwards'] = playerStats19.groupby(['Squad'])['Pos'].apply(lambda x: x[x.str.contains('FW')].count())
playerStats19Team['Goalkeeper'] = playerStats19.groupby(['Squad'])['Pos'].apply(lambda x: x[x.str.contains('FW')].count())
playerStats19Team['Defend & Midfield'] = playerStats19.groupby(['Squad'])['Pos'].apply(lambda x: x[x.str.contains('DF,MF')].count())
playerStats19Team['Forwards & Midfield'] = playerStats19.groupby(['Squad'])['Pos'].apply(lambda x: x[x.str.contains('FW,MF')].count())




playerStats20Team = playerStats20.groupby(by=["Squad"]).sum()
playerStats20Team['Defender'] = playerStats20.groupby(['Squad'])['Pos'].apply(lambda x: x[x.str.contains('DF')].count())
playerStats20Team['Midfielders'] = playerStats20.groupby(['Squad'])['Pos'].apply(lambda x: x[x.str.contains('MF')].count())
playerStats20Team['Forwards'] = playerStats20.groupby(['Squad'])['Pos'].apply(lambda x: x[x.str.contains('FW')].count())
playerStats20Team['Goalkeeper'] = playerStats20.groupby(['Squad'])['Pos'].apply(lambda x: x[x.str.contains('FW')].count())
playerStats20Team['Defend & Midfield'] = playerStats20.groupby(['Squad'])['Pos'].apply(lambda x: x[x.str.contains('DF,MF')].count())
playerStats20Team['Forwards & Midfield'] = playerStats20.groupby(['Squad'])['Pos'].apply(lambda x: x[x.str.contains('FW,MF')].count())




playerStats17Team_mean = playerStats17.groupby(by=["Squad"]).mean()
playerStats18Team_mean = playerStats18.groupby(by=["Squad"]).mean()
playerStats19Team_mean = playerStats19.groupby(by=["Squad"]).mean()
playerStats20Team_mean = playerStats20.groupby(by=["Squad"]).mean()

playerStats17Team_median = playerStats17.groupby(by=["Squad"]).mean()
playerStats18Team_median = playerStats18.groupby(by=["Squad"]).mean()
playerStats19Team_median = playerStats19.groupby(by=["Squad"]).mean()
playerStats20Team_median = playerStats20.groupby(by=["Squad"]).mean()






player_statistic_grouped = pd.DataFrame({
    'Median Age 17 med': playerStats17Team_median['Age'],
    'Median Age 18 med': playerStats18Team_median['Age'],
    'Median Age 19 med': playerStats19Team_median['Age'],
    'Median Age 20 med': playerStats20Team_median['Age'],
    
    'Matches Played 17 med': playerStats17Team_median['MP'],
    'Matches Played 18 med': playerStats18Team_median['MP'],
    'Matches Played 19 med': playerStats19Team_median['MP'],
    
    'Goals 90min 17 mean': playerStats17Team_mean['Gls'],
    'Goals 90min 18 mean': playerStats18Team_mean['Gls'],
   'Goals 90min 19 mean': playerStats19Team_mean['Gls'],
   
   'Penalty goals 17': playerStats17Team_mean['PK'],
   'Penalty goals 18': playerStats18Team_mean['PK'],
   'Penalty goals 19': playerStats19Team_mean['PK'],
    
    
    'Yellow Cards mean 17': playerStats17Team_mean['CrdY'],
    'Yellow Cards mean 18': playerStats18Team_mean['CrdY'],
    'Yellow Cards mean 18': playerStats19Team_mean['CrdY'],
    
    'Red Cards mean 17': playerStats17Team_mean['CrdR'],
    'Red Cards mean 18': playerStats18Team_mean['CrdR'],
    'Red Cards mean 19': playerStats19Team_mean['CrdR'],
    
    'Defender 17': playerStats17Team['Defender'],
    'Midfielders 17': playerStats17Team['Midfielders'],
    'Forwards 17': playerStats17Team['Forwards'],
    'Goealkeeper 17': playerStats17Team['Goalkeeper'],
    'Defend & Midfield 17': playerStats17Team['Defend & Midfield'],
    'Forwards & Midfield 17': playerStats17Team['Forwards & Midfield'],
    
    'Defender 18': playerStats18Team['Defender'],
    'Midfielders 18': playerStats18Team['Midfielders'],
    'Forwards 18': playerStats18Team['Forwards'],
    'Goealkeeper 18': playerStats18Team['Goalkeeper'],
    'Defend & Midfield 18': playerStats18Team['Defend & Midfield'],
    'Forwards & Midfield 18': playerStats18Team['Forwards & Midfield'],
    
    'Defender 19': playerStats19Team['Defender'],
    'Midfielders 19': playerStats19Team['Midfielders'],
    'Forwards 19': playerStats19Team['Forwards'],
    'Goealkeeper 19': playerStats19Team['Goalkeeper'],
    'Defend & Midfield 19': playerStats19Team['Defend & Midfield'],
    'Forwards & Midfield 19': playerStats19Team['Forwards & Midfield'],
    
    'Defender 20': playerStats20Team['Defender'],
    'Midfielders 20': playerStats20Team['Midfielders'],
    'Forwards 20': playerStats20Team['Forwards'],
    'Goealkeeper 20': playerStats20Team['Goalkeeper'],
    'Defend & Midfield 20': playerStats20Team['Defend & Midfield'],
    'Forwards & Midfield 20': playerStats20Team['Forwards & Midfield'],


    
    

})

correlation17 = player_statistic_grouped.iloc[:, [0, 7, 10, 15, 18, 19, 20, 21, 22, 23]].corr()
fig1 = go.Figure(data=go.Heatmap(z=correlation17, x=correlation17.columns, y =correlation17.columns))
fig1.update_layout(title_text='Correlation plot for particular statistic for teams 2017 example')
fig1.show()

# %%


table17 = pd.read_html('2017/table.xls', encoding='utf-8')[0]
table18 = pd.read_html('2018/table.xls', encoding='utf-8')[0]
table19 = pd.read_html('2019/table.xls', encoding='utf-8')[0]
#table20 = pd.read_html('2020/table.xls')[0]




table17[['Top Team Scorer', 'Top Team Scorer Goals']] = table17['Top Team Scorer'].str.split(' - ', 1, expand=True)
table18[['Top Team Scorer', 'Top Team Scorer Goals']] = table18['Top Team Scorer'].str.split(' - ', 1, expand=True)
table19[['Top Team Scorer', 'Top Team Scorer Goals']] = table19['Top Team Scorer'].str.split(' - ', 1, expand=True)
#table20[['Top Team Scorer', 'Top Team Scorer Goals']] = table20['Top Team Scorer'].str.split(' - ', 1, expand=True)


figS = px.scatter(
        table17, x="Squad", y="GF", 
        size =list(map(int, table17['Top Team Scorer Goals'])),  color=list(map(int, table17['Top Team Scorer Goals'])), 
        hover_data=['Top Team Scorer'])

figS.update_layout(title_text='Goals for teams in 2017 with goals scored by the best scorer in the team')
figS.show()



y17 = list(map(int, table17['GF']))
x17 = table17['Squad']
x17 = x17.values.tolist()


y18 = list(map(int, table18['GF']))
x18 = table18['Squad']
x18 = x18.values.tolist()


y19 = list(map(int, table19['GF']))
x19 = table19['Squad']
x19 = x19.values.tolist()



fig2 = go.Figure(data=[
    go.Bar(name='2017', x=x17, y=y17),
    go.Bar(name='2018', x=x18, y=y18),
    go.Bar(name='2019', x=x19, y=y19)

])
fig2.update_layout(title_text='Goals for teams 2017-2019')

#fig2.show()

df = table17.corr()
fig = go.Figure(data=go.Heatmap(z=df, x=df.columns, y =df.columns))
#fig.show()

# %%
teamStats17 = pd.read_html('2017/team-stats.xls', encoding='utf-8',header=1)[0]
teamStats18 = pd.read_html('2018/team-stats.xls', encoding='utf-8',header=1)[0]
teamStats19 = pd.read_html('2019/team-stats.xls', encoding='utf-8',header=1)[0]
#teamStats20 = pd.read_html('2020/team-stats.xls'), encoding='utf-8',header=1)[0]

teamStats17 = teamStats17.dropna(axis=1 , how='all').dropna(axis=0 , how='all')
teamStats18 = teamStats18.dropna(axis=1, how='all').dropna(axis=0 , how='all')
teamStats19 = teamStats19.dropna(axis=1, how='all').dropna(axis=0 , how='all')

teamStats17 = teamStats17.iloc[:, [0, 1,2, 7, 8, 9, 10, 11, 12, 13, 14]]
teamStats18 = teamStats18.iloc[:, [0, 1,2, 7, 8, 9, 10, 11, 12, 13, 14]]
teamStats19 = teamStats19.iloc[:, [0, 1,2, 7, 8, 9, 10, 11, 12, 13, 14]]
#teamStats20 = teamStats18.iloc[:, [0, 1,2, 7, 8, 9, 10, 11, 12, 13, 14]]

