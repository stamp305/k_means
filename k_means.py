# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 2025
@author: YourName
Description: Simple k-Means Clustering Visualization using synthetic data
"""
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

st.title("🔍 K-Means Clustering Visualizer")
st.write("Example Data for Visualization")

# Generate synthetic dataset
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Apply k-means clustering
kmeans = KMeans(n_clusters=4, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', s=50)
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
           s=300, c='red', marker='X', label='Centroids')
ax.set_title('k-Means Clustering')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.legend()
ax.grid(True)
st.pyplot(fig)
