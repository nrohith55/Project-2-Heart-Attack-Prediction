# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 01:07:05 2020

@author: Rohith
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.metrics import cohen_kappa_score

df=pd.read_csv("E:\\Data Science\\Project_3\\data.csv",na_values=['?'])


print(df)

#############################Data Exploration#################################

df.head() # To show first 5 rows of the data set

df.tail()#to show last 5 columns of the data set

df.describe() #Describe the data set

df.columns#To show number columns

df.dtypes ##shows what type the data was read in as (float, int, string, bool, etc.)

df.shape


df.isnull().sum()#To find is thr any na values

#Imputation is the process of replacing the missing values

df.trestbps.mode()  # 120 
df["trestbps"].fillna(120,inplace=True)

df.chol.mode() # 275
df["chol"].fillna(275,inplace=True)

df.fbs.mode() #0
df['fbs'].fillna(0,inplace=True)

df.restecg.mode()  #0
df["restecg"].fillna(0,inplace=True)

df.thalach.mode() # 150
df["thalach"].fillna(150,inplace=True)

df.exang.mode() #0
df['exang'].fillna(0,inplace=True)

df.slope.mode()#2
df['slope'].fillna(2,inplace=True)

df.ca.mode() #0
df['ca'].fillna(0,inplace=True)

df.thal.mode()#7
df['thal'].fillna(7,inplace=True)



df.shape # Before dropping duplicatres

df.drop_duplicates(inplace=True)#Removing the duplicate values

df.shape #After dropping duplicates


############################Data vizualization###########################################################

df.hist(figsize=(12,12)) # histogram plots

df.plot(kind='box',subplots=True,layout=(4,4),sharex=False,sharey=False,figsize=(18,18)) # Box plot

df.corr()#To find the correlation

sns.pairplot(df) # To plot pairs plot

#Correlation matrix
cormat=df.corr()
fig= plt.figure(figsize=(12,12))
sns.heatmap(cormat,annot=True,cmap="BuGn_r")
plt.show()

##########Since we have sex','cp','fbs','restecg','exang','slope','ca','thal' these are in categorical values we have to covert into dummy values"###
##...need to convert some variable to dummy variable....#
df=pd.get_dummies(df,columns=['sex','cp','fbs','restecg','exang','slope','ca','thal'],drop_first=True)

################### Model Building step ####################################################################

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

n_errors_Log=(y_pred!=y_test).sum()
print(n_errors_Log)#7
cohen_kappa_score(y_test,y_pred)#0.7244

############################################################################################################


from sklearn.tree import DecisionTreeClassifier #Decision tree classifier

model1=DecisionTreeClassifier(criterion='entropy')
model1.fit(X_train,y_train)
y_pred1=model1.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(y_test,y_pred1))#   0.75
print(confusion_matrix(y_test,y_pred1))
print(classification_report(y_test,y_pred1))
n_errors_dec=(y_pred1!=y_test).sum()
print(n_errors_dec)#15

cohen_kappa_score(y_test,y_pred1)#0.39
#######################################################################################################

from sklearn.ensemble import RandomForestClassifier #Random forest classifier

model2=RandomForestClassifier(n_estimators=30)
model2.fit(X_train,y_train)
y_pred2=model2.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print(accuracy_score(y_test,y_pred2))# 0.83
print(confusion_matrix(y_test,y_pred2))
print(classification_report(y_test,y_pred2))
n_errors_ran=(y_pred2!=y_test).sum()
print(n_errors_ran)#11
cohen_kappa_score(y_test,y_pred2)#0.61
########################################################################################################

from sklearn.ensemble import ExtraTreesClassifier #Extratreeclassifier

model3=ExtraTreesClassifier()
model3.fit(X_train,y_train)
y_pred3=model3.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print(accuracy_score(y_test,y_pred3))#  0.83
print(confusion_matrix(y_test,y_pred3))
print(classification_report(y_test,y_pred3))
n_errors_ext=(y_pred3!=y_test).sum()
print(n_errors_ext)#10

cohen_kappa_score(y_test,y_pred3) #0.6237

#########################################################################################################################################################

from sklearn.svm import SVC  ##################### SVM##################

model4=SVC()
model4.fit(X_train,y_train)

y_pred4=model4.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(y_test,y_pred4))#0.86
print(confusion_matrix(y_test,y_pred4))
print(classification_report(y_test,y_pred4))

n_errors_svc=(y_pred4!=y_test).sum()
print(n_errors_svc)#8


cohen_kappa_score(y_test,y_pred4)#0.68021
###############################################################################################################################################################

from sklearn.neural_network import MLPClassifier ###############Neural_Networks############

model5=MLPClassifier(hidden_layer_sizes=(5,5))

model5.fit(X_train,y_train)

y_pred5=model5.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(y_test,y_pred5))#     0.81  
print(confusion_matrix(y_test,y_pred5))
print(classification_report(y_test,y_pred5))

n_errors_nn=(y_pred5!=y_test).sum()
print(n_errors_nn)#11

cohen_kappa_score(y_test,y_pred5)#0.56
##############################################################################################################################################################


from sklearn.ensemble import BaggingClassifier #######################Bagging Classifier

model6=BaggingClassifier(DecisionTreeClassifier(criterion='entropy'))
model6.fit(X_train,y_train)

y_pred6=model6.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(y_test,y_pred6))#0.74
print(confusion_matrix(y_test,y_pred6))
print(classification_report(y_test,y_pred6))


n_errors_BC=(y_pred6!=y_test).sum()
print(n_errors_BC)#15

cohen_kappa_score(y_test,y_pred6) #0.39

############################################################################################################################################################

#So from the above all model building results we have kappa Score of 0.72 
#predicting errors =7 accuracy is 88.13 which is good in Logistic regression.
#So the final model is Logistic regression






















































