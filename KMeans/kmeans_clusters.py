import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
from sklearn.datasets._samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin

# cluster_std: standard deviation
# random_state: random seed
# X contains the coordinates of the generated points
# y_true contains the true cluster labels of the points
X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)
# X[:, 0], X[:, 1]: extract the first and second columns
# s: the size of the markers in the scatter plot
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.show()

# scikit-learn version
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# cmap: use the Viridis color map for the plot
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
# cluster_centers_: obtain the coordinates of the centroids of the 4 clusters
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()

# Expectation–Maximization version
# 1. Guess some cluster centers
# 2. Repeat until converged
#   1. E-Step: assign points to the nearest cluster center
#   2. M-Step: set the cluster centers to the mean
def find_clusters(X, n_clusters, rseed=2):
    # 1. Randomly choose clusters

    # RandomState: used to create a random number generator with a set seed
    # It is useful because using seed() impacts the global numpy random generation, 
    # while RandomState will set the seed for the rng generator only
    rng = np.random.RandomState(rseed)
    # rng.permutation(X.shape[0]): creates a random permutation of the integers from 0 to X.shape[0]-1
    # [:n_clusters]: selects the first n_clusters integers from the permutation
    # i: selected indices 
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    
    while True:
        # 2a. Assign labels based on closest center
        # pairwise_distances_argmin：Compute minimum distances between one point and a set of points.
        labels = pairwise_distances_argmin(X, centers)
        
        # 2b. Find new centers from means of points
        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])
        
        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers
    
    return centers, labels

centers, labels = find_clusters(X, 4)
plt.scatter(X[:, 0], X[:, 1], c=labels,
            s=50, cmap='viridis');
plt.show()