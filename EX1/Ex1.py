#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 09:42:40 2021

@author: edyta
"""

import pandas as pd
#Exercise 1
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

#Exercise 2 (The ramen king)
ramen = pd.read_csv("ramen-ratings.csv") 
summary_country= ramen.groupby(['Country']).count()
norm = ramen.groupby(['Country', 'Style']).apply(lambda x: x['Review #']/x['Review #'].sum())

#Exercise 3
orders = pd.read_csv("orders.csv") 
customers = pd.read_csv("customers.csv") 


customers= customers[customers.columns[0:2]]
orders= orders[orders.columns[1:3]]

customers['gender'] = (customers['gender'].str.upper())
customers['gender'].str.strip('gender')
customers['gender'].dropna()
orders['item_count'].dropna()

customers.drop(index=customers[customers['gender'] == 'nan'].index, inplace=True)

customers.drop(index=customers[customers['gender'] == '?????'].index, inplace=True)
customers.drop(index=customers[customers['gender'] == '  '].index, inplace=True)
customers.drop(index=customers[customers['gender'] == ''].index, inplace=True)


joined = orders.join(customers.set_index('akeed_customer_id'), on='customer_id')
joined.drop(index=joined[joined['gender'] == 'nan'].index, inplace=True)
df = joined
nan_value = float("NaN")

df.replace("", nan_value, inplace=True)
df.dropna(subset = ["gender"], inplace=True)
df.dropna(subset = ["item_count"], inplace=True)
points = df.gender.unique()

df['gender'] = df['gender'].str.replace('FEMALE','F')
df['gender'] = df['gender'].str.replace('MALE','M')

df['gender'] = df['gender'].apply(lambda x:x.strip() )

statistikk = df.groupby(['gender'])['item_count'].mean()
