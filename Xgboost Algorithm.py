# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 19:16:30 2020

@author: Shrikant Agrawal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

""" In this dataset we have many variables. From CreditScores to EstimatedSalary are our 
independent variables and last column Exited will be our dependent variable. It defines wheather 
customer has left the bank or not.

You can't see X variable data in the variable explorer because it has many datatypes. 
To view press X enter in the console"""
 
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

"""Encoding categorical data - In X we have different columns out of which two columns
Georgrapy and Gender are categorical variables which are on number 1 and 2. 
To convert it to numerical we are performing below steps"""

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])  

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting XGBoost to the Training set
import xgboost
classifier = xgboost.XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
# Applying k-Fold Cross Validation to check accuracy scores
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

# OR
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
accuracy