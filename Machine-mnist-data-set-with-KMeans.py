import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import rand_score
from sklearn.datasets import make_blobs
np.random.seed(42)

# ------------------------------------------------------------------------------
# helper function for calculate euclidean distance
# ------------------------------------------------------------------------------

def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# ------------------------------------------------------------------------------
# Implement of the k-means algorithm
# ------------------------------------------------------------------------------
class KMeans:
    def __init__(self, K = 5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]

        # mean feature vector for each cluster
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape
        # print(self.n_samples)
        # print(self.n_features)

        # initilize centtroids
        # make sure we don't pick same point twice
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False) # array of size k
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # opzimazation
        for _ in range(self.max_iters):
            # update clusters
            self.clusters = self._create_clusters(self.centroids)

            # #*************
            # if self.plot_steps:
            #     self.plot()
            # #************

            # update centroids
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            # check if converged
            if self._is_converged(centroids_old, self.centroids):
                break
            
            # #********
            # if self.plot_steps:
            #     self.plot()
            # #********

        # return cluster labels
        return self._get_cluster_labels(self.clusters)

    ## function to create clusters
    def _create_clusters(self, centroids):
        # Assign the samples to the closest centroids to create clusters
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    ## function to get the closest centroid
    def _closest_centroid(self, sample, centroids):
        # distance of the current sample to each centroid
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)# get the min distance
        return closest_index

    ## get cetroids
    def _get_centroids(self, clusters):
        # assign mean value of clusters to centroids

        # for each cluster we will store the feature vector
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            # calculate the cluster mean (cluster is the index)
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    ## check if it is converged
    def _is_converged(self, centroids_old, centroids):
        # distances between each old and new centroids, fol all centroids
        # for each cluster, calculate the distance between old and the new centroid vector
        distances = [
            euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)
        ]
        # if the sum is zero, that means no more changes, it is converaged
        return sum(distances) == 0

    ## get the labels
    def _get_cluster_labels(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        # labels here is just the index
        labels = np.empty(self.n_samples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels

    ## optional function for show plot when I use sample data
    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()


# Testing with sample 2-feature data set
X, y = make_blobs(centers=4, n_samples=100, n_features=2, shuffle=True, random_state=40)
print(X.shape)
print(y.shape)

for i in range(5, 16):
    clusters = i
    print(clusters)

    k = KMeans(K=clusters, max_iters=150, plot_steps=False)
    y_pred = k.predict(X)
    print(y_pred.shape)
    RS = rand_score(y,y_pred)
    ARS = adjusted_rand_score(y,y_pred)
    print("For k = ", k)
    print("RS: ", RS)
    print("ARS: ", ARS)
    # k.plot()
    
# ------------------------------------------------------------------------------
# Find the best k with training set
# ------------------------------------------------------------------------------
data_train = pd.read_csv("fashion-mnist_train.csv", sep = ',')
X, y = data_train.drop(['label'], axis=1),data_train['label']

X = np.array(X)
y = np.array(y)

for i in range(5, 16):
    clusters = i
    # print(clusters)

    k = KMeans(K=clusters, max_iters=150, plot_steps=False)
    y_pred = k.predict(X)
    RS = rand_score(y,y_pred)
    ARS = adjusted_rand_score(y,y_pred)
    print("\nFor k = ", i)
    print("RS: ", RS)
    print("ARS: ", ARS)


# ------------------------------------------------------------------------------
# Testing with test set
# ------------------------------------------------------------------------------
data_test = pd.read_csv("fashion-mnist_test.csv", sep = ',')
X, y = data_test.drop(['label'], axis=1),data_test['label']

X = np.array(X)
y = np.array(y)

## best k get from training set is 14 (please see the table)
clusters = 14
for i in range(4):
    k = KMeans(K=clusters, max_iters=150, plot_steps=False)
    y_pred = k.predict(X)
    RS = rand_score(y,y_pred)
    ARS = adjusted_rand_score(y,y_pred)
    print("\nFor k = ", 14)
    print("RS: ", RS)
    print("ARS: ", ARS)