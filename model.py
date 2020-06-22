# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 11:26:20 2020

@author: Rohith
"""
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.metrics import cohen_kappa_score
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

from sklearn.linear_model import LogisticRegression ########Logistic regression
model=LogisticRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(y_test,y_pred))#0.8813
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

pickle.dump(model,open('Model.pkl','wb'))
