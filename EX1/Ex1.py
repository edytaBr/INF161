#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 09:42:40 2021

@author: edyta
"""

import pandas as pd
vines = pd.read_csv("winemag-data-130k-v2.csv") 
 #step 1
prosecco = vines.loc[vines.variety == "Prosecco"]
prosecco.reset_index

#step 2
#a
df1 = prosecco.loc[(prosecco['points']>90), ['price','title', 'points']]
df2 = prosecco.loc[(prosecco['points']>84), ['price','title', 'points']]
df3 = prosecco.loc[(prosecco['points']>=84) & (prosecco['points']<=90), ['price','title', 'points']]

df1_rows = len(df1.index)
df2_rows = len(df2.index)
df3_rows = len(df3.index)

print("Number of filtered dataframes, df1, df2, df2 is equal to number of rows in prosecco df: " ,  len(prosecco.index) == (df1_rows + df2_rows + df3_rows) )

#Step3
df1["name_length"] = df1['title'].str.len()
df2["name_length"] = df2['title'].str.len()
df3["name_length"] = df3['title'].str.len()