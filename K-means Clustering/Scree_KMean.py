# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 17:53:57 2022

@author: Renee
"""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns 
from sklearn.metrics import silhouette_score

df = pd.read_csv(r'C:/Users/reneeh2/OneDrive - University of Illinois - Urbana/Desktop/IS 517/UCI Dataset_Medians.csv')             
#print (df)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(df)
    wcss.append(kmeans.inertia_)

sns.set()
plt.plot(range(1, 11), wcss)
plt.title('Selecting the Number of Clusters using the Elbow Method')
plt.xlabel('Clusters')
plt.ylabel('Within Cluster Sum of Squares')
plt.show()

kmeans = KMeans(n_clusters=5).fit(df)
print("Cluster Centroid Values:")
centroids = kmeans.cluster_centers_
print(centroids)
label = kmeans.fit_predict(df)
#print(label, df)
kmeans_silhouette = silhouette_score(df, label)
print("Overall Average Silhouette Score:")
print(kmeans_silhouette)

plt.scatter(df['Age'], df['total_UPDRS'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.xlabel('Age')
plt.ylabel('total_UPDRS')
plt.show()