import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.5)
n_clusters = 4
max_iteration = 300
plt.scatter(X[:,0], X[:,1])
plt.show()

def initialize_centroid(X):
    random_idx = np.random.permutation(X.shape[0])
    centroids = X[random_idx[:n_clusters]]
    return centroids

def assign_cluster(X, centroids):
    distance = np.zeros((X.shape[0], n_clusters))
    for k in range(n_clusters):
        row_norm = np.linalg.norm(X - centroids[k, :], axis=1)
        distance[:, k] = np.square(row_norm)
        
    cluster_label = np.argmin(distance, axis=1)
    return cluster_label

def compute_centroids(X, labels):
    centroids = np.zeros((n_clusters, X.shape[1]))
    for k in range(n_clusters):
        centroids[k, :] = np.mean(X[labels == k, :], axis=0)
    return centroids

def compute_sse(X, labels, centroids):
    distance = np.zeros(X.shape[0])
    for k in range(n_clusters):
        distance[labels == k] = np.linalg.norm(X[labels == k] - centroids[k], axis=1)
    return np.sum(np.square(distance))

sse = []
cent = []
for j in range(10):
    new_centroids = initialize_centroid(X)
    for i in range(max_iteration):
        old_centroids = new_centroids
        labels = assign_cluster(X, old_centroids)
        new_centroids = compute_centroids(X, labels)
        if np.all(old_centroids == new_centroids):
            break
    sse.append(compute_sse(X,labels, new_centroids))
    cent.append(new_centroids)

plt.scatter(X[:,0], X[:,1])
plt.scatter(cent[np.argmin(sse)][:, 0], cent[np.argmin(sse)][:, 1], s=300, c='red')
plt.show()