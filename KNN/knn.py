import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter

# In practice, k is usually chosen at random between 3 and 10.
# A small value of k results in unstable decision boundaries. 
# A large value of k often leads to the smoothening of decision boundaries but not always to better metrics.

# center_scale: scale of cluster centroid, standard deviation
# cluster_scale: scale of individual cluster, standard deviation
# class_counts: determines the size of each cluster (40)
def generate_data(center_scale: float, cluster_scale: float, class_counts: np.ndarray,
                  seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    # Fix a seed to make experiment reproducible
    np.random.seed(seed)
    # points: a numpy array of shape (N, 2) that contains the generated data points.
    # first column is x value of each point
    # second column is y value of each point
    # classes: a numpy array of shape (N,) that contains the corresponding class labels for each point
    # the labels are int from 0 to (len(class_counts) - 1)
    points, classes = [], []
    # class_index corresponds to the index of the current element in class_counts, 
    # and class_count corresponds to the value of the current element. 
    for class_index, class_count in enumerate(class_counts):
        # Generate the center of the cluster and its points centered around it
        # a random point (as a 2D array) drawn from a normal distribution with standard deviation center_scale.
        current_center = np.random.normal(scale=center_scale, size=(1, 2))
        # (+ cunrrent_cnter) can help move the cluster to center around centroid point, current_center
        current_points = np.random.normal(scale=cluster_scale, size=(class_count, 2)) + current_center
        # Assign them to the same class and add those points to the general pool
        # output: [0 0 ... 0] [1 1 ... 1] [2 2 ... 2]
        # size of each array is class_count
        current_classes = np.ones(class_count, dtype=np.int64) * class_index
        points.append(current_points)
        classes.append(current_classes)
    # Concatenate clusters into a single array of points
    points = np.concatenate(points)
    classes = np.concatenate(classes)
    return points, classes

points, classes = generate_data(2, 0.75, [40, 40, 40], seed=42)

# bmh: Bayesian Methods for Hackers (pre-defined style)
plt.style.use('bmh')

def plot_data(points: np.ndarray, classes: np.ndarray) -> None:
    # figsize: size; dpi: resolution
    # fig is the entire figure, while ax is a single subplot within the figure
    # This allows us to customize the properties of the scatter plot (e.g., the colors, markers, etc.) on a per-subplot basis.
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    scatter = ax.scatter(x=points[:, 0], y=points[:, 1], c=classes, cmap='prism', edgecolor='black')
    # Generate a legend based on the data and plot it
    legend = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
    ax.add_artist(legend)
    ax.set_title("Generated dataset")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

plot_data(points, classes)

# split training and testing data
points_train, points_test, classes_train, classes_test = train_test_split(
    points, classes, test_size=0.3
)

def classify_knn(
    points_train: np.ndarray,
    classes_train: np.ndarray,
    points_test: np.ndarray,
    num_neighbors: int
) -> np.ndarray:
    classes_test = np.zeros(points_test.shape[0], dtype=np.int64)
    for index, test_point in enumerate(points_test):
        # Compute Euclidean norm between the test point and the training dataset
        distances = np.linalg.norm(points_train - test_point, ord=2, axis=1)
        # Collect the closest neighbors indices based on the distance calculated earlier
        neighbors = np.argpartition(distances, num_neighbors)[:num_neighbors]
        # Get the classes of those neighbors and assign the most popular one to the test point
        neighbors_classes = classes_train[neighbors]
        test_point_class = Counter(neighbors_classes).most_common(1)[0][0]
        classes_test[index] = test_point_class
    return classes_test

classes_predicted = classify_knn(points_train, classes_train, points_test, 3)

