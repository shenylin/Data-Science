# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 15:36:12 2022

@author: reneeh2
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn import metrics

sentence_1 = "Humpty Dumpty sat on a wall."
sentence_2 = "Humpty Dumpty had a great fall."

CountVec = CountVectorizer(ngram_range=(1, 1), stop_words='english')

# transform
Count_data = CountVec.fit_transform([sentence_1, sentence_2])

# create dataframe
cv_dataframe = pd.DataFrame(
    Count_data.toarray(), columns=CountVec.get_feature_names_out())
print(cv_dataframe)

X = cv_dataframe
Y = np.array([0, 1])

# Create classifie
logistic_regression = LogisticRegression()
logistic_regression.fit(X, Y)
Y_pred = logistic_regression.predict(X)

#print(logistic_regression.intercept_, logistic_regression.coef_)

print('Accuracy: ', metrics.accuracy_score(Y, Y_pred))
plt.show()
