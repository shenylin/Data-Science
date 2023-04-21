# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 12:12:20 2022

@author: Renee
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

#load dataset
Admission = pd.read_csv(r'C:/Users/reneeh2/OneDrive - University of Illinois - Urbana/Desktop/IS 517/Admission.csv')
#Admission = pd.read_csv(r'C:\Users\Renee\Desktop\Admission.csv')
#print(Admission)

#Scatter Plot
plt.scatter(Admission['GPA'], Admission['GMAT'])
plt.xlabel('GPA')
plt.ylabel('GMAT')
plt.show()

#Set Features and Target Variable
X = Admission[['GPA', 'GMAT']]
Y = Admission['Decision']

#Determine classifier error rate
error_rate = []
for i in range(1,10):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X, Y)
    pred_i = knn.predict(X)
    error_rate.append(np.mean(pred_i != Y))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,10),error_rate,color='blue', linestyle='dashed', marker='o',
markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')