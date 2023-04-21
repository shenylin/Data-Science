# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 13:49:35 2022

@author: Renee
"""

#import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

#load dataset
Admission = pd.read_csv(r'C:/Users/reneeh2/OneDrive - University of Illinois - Urbana/Desktop/IS 517/Admission.csv')
#Admission = pd.read_csv(r'C:\Users\Renee\Desktop\Admission.csv')
#print(Admission)

#Set Features and Target Variable
X = Admission[['GPA', 'GMAT']]
Y = Admission['Decision']

#Split Features and Target Variables into Train and Test Sets
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=0)

#Create classifier
logistic_regression= LogisticRegression(penalty='l1', solver='liblinear')
#logistic_regression= LogisticRegression(penalty='l2', solver='liblinear')
logistic_regression.fit(X_train,Y_train)
Y_pred=logistic_regression.predict(X_test)

print('Coefficients', logistic_regression.intercept_, logistic_regression.coef_)

confusion_matrix = pd.crosstab(Y_test, Y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)

print('Accuracy: ',metrics.accuracy_score(Y_test, Y_pred))
plt.show()

#print (X_test)
#print (Y_pred)

print(classification_report(Y_test, Y_pred))