# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 12:36:29 2022

@author: Renee
"""

#import libraries
import pandas as pd
#from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import tree

#load dataset
Iris = pd.read_csv(r'C:/Users/reneeh2/OneDrive - University of Illinois - Urbana/Desktop/IS 517/Iris.csv')
#Iris = pd.read_csv(r'C:\Users\Renee\Desktop\iris.csv')
#print(Iris)

#Set Features and Target Variable
X = Iris[['x1', 'x2', 'x3', 'x4']]
Y = Iris['type']

#Create classifier, number of trees, n
clf = DecisionTreeClassifier(max_depth=4)
#clf= RandomForestClassifier(n_estimators=10)
clf.fit(X,Y)
Y_pred=clf.predict(X)

confusion_matrix = pd.crosstab(Y, Y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)

print('Accuracy: ',metrics.accuracy_score(Y, Y_pred))
plt.show()

#print (X)
#print (Y_pred)

#print(classification_report(Y, Y_pred))

importances = list(clf.feature_importances_)
# Print out the feature and importances 
print (importances)

fig = plt.figure(figsize=(12, 10))
tree.plot_tree(clf.fit(X,Y), class_names=['Type 1', 'Type 2', 'Type 3'])
plt.show()

