#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 12:48:50 2021

@author: edyta
"""

import pandas as pd
import matplotlib.pyplot as plt
import codecs
from pyecharts.charts import Bar
from pyecharts import options as opts
from pyecharts.globals import ThemeType
import plotly.express as px
import plotly.graph_objects as go


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


playerStats17Team = playerStats17.groupby(by=["Squad"]).sum()
playerStats18Team = playerStats18.groupby(by=["Squad"]).sum()
playerStats19Team = playerStats19.groupby(by=["Squad"]).sum()
playerStats20Team = playerStats20.groupby(by=["Squad"]).sum()


#Unified nation
playerStats17['Nation'] = playerStats17['Nation'].str[-3:]
playerStats18['Nation'] = playerStats18['Nation'].str[-3:]
playerStats19['Nation'] = playerStats19['Nation'].str[-3:]
playerStats20['Nation'] = playerStats20['Nation'].str[-3:]

df = playerStats17

#After checking what teams we do have each season by playerStats(year).Squad.unique() I see that it varries. Each year we have new teams that ar not in ranking and some new.
bar = (Bar(init_opts=opts.InitOpts(theme=ThemeType.PURPLE_PASSION)).add_xaxis(['a', 'a']).
       add_yaxis("Temperature Max", [-7,-6,-2,4,10,15,18,17,13,7,2,-3])
       .add_yaxis("Temperature Min", [-1,0,5,12,18,24,27,26,21,14,8,2]).set_global_opts(title_opts=opts.TitleOpts(title="30-year temperature for Toronto", subtitle="Year 1981 to 2010")))
bar.render_notebook()


# %%


table17 = pd.read_html('2017/table.xls', encoding='utf-8')[0]
table18 = pd.read_html('2018/table.xls', encoding='utf-8')[0]
table19 = pd.read_html('2019/table.xls', encoding='utf-8')[0]
#table20 = pd.read_html('2020/table.xls')[0]

teamStats17 = pd.read_html('2017/team-stats.xls', encoding='utf-8')[0]
teamStats18 = pd.read_html('2018/team-stats.xls', encoding='utf-8')[0]
teamStats19 = pd.read_html('2019/team-stats.xls', encoding='utf-8')[0]
#teamStats20 = pd.read_html('2020/team-stats.xls')[0]



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

fig2.show()