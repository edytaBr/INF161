#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 12:48:50 2021

@author: edyta
"""

import pandas as pd
import matplotlib.pyplot as plt
import codecs



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


#Cleaning games
df = pd.read_html('2017/player-stats.xls', encoding='utf-8',header=1)


playerStats17 = pd.read_html('2017/player-stats.xls', encoding='utf-8',header=1)[0]
playerStats18 = pd.read_html('2018/player-stats.xls', encoding='utf-8', header=1)[0]
playerStats19 = pd.read_html('2019/player-stats.xls', encoding='utf-8', header=1)[0]
playerStats20 = pd.read_html('2020/player-stats.xls', encoding='utf-8', header=1)[0]

playerStats17 = playerStats17.drop(columns=['Matches'])


#Drop nan
playerStats17 = playerStats17.dropna(axis=1 , how='all').dropna(axis=0 , how='all')
playerStats18 = playerStats18.dropna(axis=1, how='all').dropna(axis=0 , how='all')
playerStats19 = playerStats19.dropna(axis=1, how='all').dropna(axis=0 , how='all')
playerStats20 = playerStats20.dropna(axis=1, how='all').dropna(axis=0 , how='all')










# %%












table17 = pd.read_html('2017/table.xls', encoding='utf-8')[0]
table18 = pd.read_html('2018/table.xls', encoding='utf-8')[0]
table19 = pd.read_html('2019/table.xls', encoding='utf-8')[0]
#table20 = pd.read_html('2020/table.xls')[0]

teamStats17 = pd.read_html('2017/team-stats.xls', encoding='utf-8')[0]
teamStats18 = pd.read_html('2018/team-stats.xls', encoding='utf-8')[0]
teamStats19 = pd.read_html('2019/team-stats.xls', encoding='utf-8')[0]
#teamStats20 = pd.read_html('2020/team-stats.xls')[0]





