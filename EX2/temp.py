#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 17:00:27 2021

@author: edyta
"""
import pandas as pd  # To read data
import numpy as np
import statistics
import scipy.stats as st


data = [45, 70, 94, 78, 61, 18, 19, 34, 48, 73, 56, 46, 32, 95, 47, 39, 96, 58, 1, 23, 30, 50, 21, 47, 80, 23, 38, 33, 5, 39]

df = pd.DataFrame(data,columns=['Val'])
a = st.t.interval(alpha=0.9, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)) 
