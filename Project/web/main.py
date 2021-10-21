#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 11:07:29 2021

@author: edyta
"""

from flask import Flask, render_template
import pandas as pd
app = Flask(__name__)
@app.route("/")

def template():
   data = pd.read_csv("results.csv") 
   myData = data.values
   return render_template("template.html", myData=myData)


if __name__ == "__main__":
  app.run(debug=True)