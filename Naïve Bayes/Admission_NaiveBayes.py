# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 14:54:48 2022

@author: Renee
"""

#import libraries
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

#load dataset
Admission = pd.read_csv(r'C:/Users/reneeh2/OneDrive - University of Illinois - Urbana/Desktop/IS 517/Admission.csv')
#Admission = pd.read_csv(r'C:\Users\Renee\Desktop\Admission.csv')
print(Admission)

#Scatter Plot
#plt.scatter(Admission['GPA'], Admission['GMAT'])
#plt.xlabel('GPA')
#plt.ylabel('GMAT')
#plt.show()

#Set Features and Target Variable
X = Admission[['GPA', 'GMAT']]
Y = Admission['Decision']

# Create classifier
nv = GaussianNB() 
nv.fit(X,Y)

Y_pred = nv.predict(X) # store the prediction data
accuracy_score(Y,Y_pred) # calculate the accuracy

confusion_matrix = pd.crosstab(Y, Y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)

print('Accuracy: ',metrics.accuracy_score(Y, Y_pred))
plt.show()

print (X)
print (Y_pred)

#prediction = nv.predict([[3.78,591]]) 
#print ('Predicted Result: ', prediction)

#print(classification_report(Y, Y_pred))