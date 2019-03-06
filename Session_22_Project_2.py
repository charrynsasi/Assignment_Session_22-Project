# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 13:56:01 2019

@author: CNsasi
"""
#importing libraries
import sqlite3
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#importing the dataset
cnx = sqlite3.connect('database.sqlite')
df = pd.read_sql_query("SELECT * FROM Player_Attributes", cnx)

#removing unnecessary column
df.drop(['id','player_fifa_api_id','player_api_id','date'],axis=1,inplace=True)

#Split our dependant and independant feature
X=df.iloc[:,1:]
y=df.iloc[:,0]

#Encoding categorical feature
preferred_foots=pd.get_dummies(X['preferred_foot'], drop_first=True)
attacking_work_rates=pd.get_dummies(X['attacking_work_rate'], drop_first=True)
defensive_work_rates=pd.get_dummies(X['defensive_work_rate'], drop_first=True)

#Drop the columns
X.drop(['preferred_foot','attacking_work_rate','defensive_work_rate'],axis=1,inplace=True) 

#Taking care of mising data on our independant feature
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer=imputer.fit(X.iloc[:,:])
X.iloc[:,:]=imputer.transform(X.iloc[:,:])

#Taking care of mising data on our dependant feature
y = y.fillna(y.mean())

#concat the dummy variables
X=pd.concat([X,preferred_foots,attacking_work_rates,defensive_work_rates], axis=1)

#splitting data into training and test set
X_train, X_test, y_train, y_test =train_test_split(X,y, test_size=0.3, random_state=0)

#fitting multiple linear regresion to the training test
regressor=LinearRegression()
regressor.fit(X_train, y_train)

#predicting test set
y_pred=regressor.predict(X_test)

#measure accuracy
score=r2_score(y_test,y_pred)


