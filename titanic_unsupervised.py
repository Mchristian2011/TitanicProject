"""
Author: Your Name
Date: 09/07/2025
Course: Machine Learning
Description: Unsupervised clustering on Titanic dataset using KMeans.
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load Titanic dataset
data = pd.read_csv("titanic.csv")

# Use selected features
features = data[['Age', 'Fare', 'Pclass']].dropna()

# Scale the data
scaler = StandardScaler()
scaled = scaler.fit_transform(features)

# Apply KMeans with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(scaled)

# Add clusters to data
features['Cluster'] = clusters

# Show sample results
print(features.head(10))
print("\nCluster distribution:")
print(features['Cluster'].value_counts())
