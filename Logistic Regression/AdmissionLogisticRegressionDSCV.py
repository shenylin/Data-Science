# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 17:36:05 2022

@author: reneeh2
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 13:49:35 2022

@author: Renee
"""

#import libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
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

#Create classifier
logistic_regression= LogisticRegression(penalty='l1', solver='liblinear')
#logistic_regression= LogisticRegression(penalty='l2', solver='liblinear')
logistic_regression.fit(X,Y)
Y_pred=logistic_regression.predict(X)

#results = cross_val_score(logistic_regression, X, Y, cv=kfold)
results = cross_val_score(logistic_regression, X, Y, cv=5)
print('Cross-Validation Accuracy', results)

print('Coefficients', logistic_regression.intercept_, logistic_regression.coef_)

confusion_matrix = pd.crosstab(Y, Y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)

print('Accuracy: ',metrics.accuracy_score(Y, Y_pred))
plt.show()

print(classification_report(Y, Y_pred))

