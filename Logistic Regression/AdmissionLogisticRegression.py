# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 14:29:17 2022

@author: Renee
"""

#import libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

#load dataset
Admission = pd.read_csv(r'C:/Users/reneeh2/OneDrive - University of Illinois - Urbana/Desktop/IS 517/Admission.csv')
#print(Admission)

#Scatter Plot
plt.scatter(Admission['GPA'], Admission['GMAT'])
plt.xlabel('GPA')
plt.ylabel('GMAT')
plt.show()

#Set Features and Target Variable
X = Admission[['GPA', 'GMAT']]
Y = Admission['Decision']

#Create classifier
logistic_regression= LogisticRegression(penalty='l1', solver='liblinear')
#logistic_regression= LogisticRegression(penalty='l2', solver='liblinear')
logistic_regression.fit(X,Y)
Y_pred=logistic_regression.predict(X)

print(logistic_regression.intercept_, logistic_regression.coef_)

confusion_matrix = pd.crosstab(Y, Y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)

print('Accuracy: ',metrics.accuracy_score(Y, Y_pred))
plt.show()

#print (X)
print (Y_pred)

print(classification_report(Y, Y_pred))

#prediction = logistic_regression.predict([[3.78,591]]) 
#print ('Predicted Result: ', prediction)