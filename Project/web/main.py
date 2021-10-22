#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 11:07:29 2021

@author: edyta
"""

from flask import Flask, render_template
from flask import request

import pandas as pd
app = Flask(__name__)
@app.route("/")

def template():

   return render_template("template.html")




@app.route('/', methods=['POST'])
def score_send():
    data = pd.read_csv("results.csv") 

    h = request.form['home']
    a = request.form['away']
    model = request.form['model']
    
    for elem in data.itertuples():
        if ((model == 'knn') and ((elem.Home == h) and (elem.Away == a))):
            score_Home = elem.HomeScore_KN
            score_Away = elem.AwayScore_KN
        elif ((model == 'randomforest') and ((elem.Home == h) and (elem.Away == a))):
            score_Home = elem.HomeScore_RF
            score_Away = elem.AwayScore_RF
            
    return render_template("predict.html", myData = [ score_Home, score_Away])





if __name__ == "__main__":
  app.run(debug=True)