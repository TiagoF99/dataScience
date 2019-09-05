"""
ALGORITHM STEPS


1. Initisalize k centroids randomly
2. calculate distance for each point to group
3. assign each point to closest centroid and you have clusters
4. calculate new centroids for each cluster found in 3. by getting average of
   each point in each cluster.
5. repeat steps 2. 3.
6. repeat 4 and 5 until there are no more changes to be made (iteratively)

Measuring Accuracy:

1. external: compare clusters with the ground truth
2. internal: average the distance between data points within a cluster.

How to find k:
run it on many points and calculate the accuracy. Then take the point where
the decrease rapidly shifts (elbow point) >> (So graph k on x-axis by the
mean distance of data points to each centroid on y-axis)
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

# making our own data
X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)
plt.scatter(X[:, 0], X[:, 1], marker='.')
plt.show()

"""
initialize k-means algorithm:
The KMeans class has many parameters that can be used, but we will be using 
these three:
    init: Initialization method of the centroids.
        - Value will be: "k-means++"
        - k-means++: Selects initial cluster centers for k-mean clustering in a 
        smart way to speed up convergence.
        
    n_clusters: The number of clusters to form as well as the number of 
    centroids to generate.
       -  Value will be: 4 (since we have 4 centers)
        
    n_init: Number of time the k-means algorithm will be run with different 
    centroid seeds. The final results will be the best output of n_init 
    consecutive runs in terms of inertia.
        - Value will be: 12
"""

k_means = KMeans(init="k-means++", n_clusters=4, n_init=12)
k_means.fit(X)
# says what cluster (0,1,2,3) each point is in
k_means_labels = k_means.labels_
# coordinates of each cluster center
k_means_cluster_centers = k_means.cluster_centers_

"""
Visulaize Data
"""
# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(6, 4))
# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the
# unique labels.
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))
# Create a plot
ax = fig.add_subplot(1, 1, 1)

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.
for k, col in zip(range(len([[4, 4], [-2, -1], [2, -3], [1, 1]])), colors):
    # Create a list of all data points, where the data poitns that are
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_means_labels == k)
    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]
    # Plots the datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col,
            marker='.')
    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)

# Title of the plot
ax.set_title('KMeans')
# Remove x-axis ticks
ax.set_xticks(())
# Remove y-axis ticks
ax.set_yticks(())
# Show the plot
plt.show()

"""
Transform to 3 clusters:
"""
k_means3 = KMeans(init = "k-means++", n_clusters = 3, n_init = 12)
k_means3.fit(X)
fig = plt.figure(figsize=(6, 4))
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means3.labels_))))
ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(len(k_means3.cluster_centers_)), colors):
    my_members = (k_means3.labels_ == k)
    cluster_center = k_means3.cluster_centers_[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)
plt.show()

"""
PRACTICAL EXAMPLE:
"""
import pandas as pd
cust_df = pd.read_csv("Cust_Segmentation.csv")
print(cust_df.head())
# drop bc its categorical data
df = cust_df.drop('Address', axis=1)

# normalize the data
from sklearn.preprocessing import StandardScaler
X = df.values[:, 1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
print(Clus_dataSet)

# Evaluate
clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
print(labels)

# assign groupings to a df column
df["Clus_km"] = labels
print(df.head(5))

# gets mean of each variable for a specific cluster
print(df.groupby('Clus_km').mean())

# distr of customers based on age and income.
area = np.pi * ( X[:, 1])**2
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)
plt.show()

# based on age, income, education
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')
ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(np.float))

"""
Next Steps:
1. Now we create a profile for each group considering common characteristics of
   each cluster. 
"""
