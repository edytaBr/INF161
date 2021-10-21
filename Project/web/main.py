#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 11:07:29 2021

@author: edyta
"""

from flask import Flask, render_template

app = Flask(__name__)
@app.route("/")

def template():
  return render_template("template.html")

@app.route("/about")
def about():
  return render_template("about.html")

@app.route("/home")
def home():
  return render_template("home.html")

if __name__ == "__main__":
  app.run(debug=True)