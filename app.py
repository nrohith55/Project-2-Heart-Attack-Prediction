# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 11:43:30 2020

@author:Rohith
"""

import pickle
from flask import Flask,render_template,url_for,request
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix # import metrics
from sklearn.model_selection import cross_val_score # import evaluation tools
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.metrics import cohen_kappa_score

model=pickle.load(open('Model.pkl', 'rb'))

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    df=pd.read_csv("E:\\Data Science\\Project_3\\data.csv",na_values=['?'])
    df["trestbps"].fillna(120,inplace=True)
    df["chol"].fillna(275,inplace=True)
    df['fbs'].fillna(0,inplace=True)
    df["restecg"].fillna(0,inplace=True)
    df["thalach"].fillna(150,inplace=True)
    df['exang'].fillna(0,inplace=True)
    df['slope'].fillna(2,inplace=True)
    df['ca'].fillna(0,inplace=True)
    df['thal'].fillna(7,inplace=True)
    df=pd.get_dummies(df,columns=['sex','cp','fbs','restecg','exang','slope','ca','thal'],drop_first=True)
    df_new=df.iloc[:,[5,0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17]]
    X=df_new.iloc[:,1:18]
    y=df_new.iloc[:,0]
    x=scale(X)
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    from sklearn.linear_model import LogisticRegression
    model=LogisticRegression()
    model.fit(X_train,y_train)
    
    if request.method == 'POST':
        my_prediction = model.predict(X_test)
        output = round(my_prediction[0], 2)
        return render_template('index.html',prediction = output)

if __name__ == '__main__':
    app.run(debug=True)