import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

def generate_data(center_scale: float, cluster_scale: float, class_counts: np.ndarray,
                  seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    # Fix a seed to make experiment reproducible
    np.random.seed(seed)
    points, classes = [], []
    for class_index, class_count in enumerate(class_counts):
        # Generate the center of the cluster and its points centered around it
        current_center = np.random.normal(scale=center_scale, size=(1, 2))
        current_points = np.random.normal(scale=cluster_scale, size=(class_count, 2)) + current_center
        # Assign them to the same class and add those points to the general pool
        current_classes = np.ones(class_count, dtype=np.int64) * class_index
        points.append(current_points)
        classes.append(current_classes)
    # Concatenate clusters into a single array of points
    points = np.concatenate(points)
    classes = np.concatenate(classes)
    return points, classes

points, classes = generate_data(2, 0.75, [40, 40, 40], seed=42)

plt.style.use('bmh')

def plot_data(points: np.ndarray, classes: np.ndarray) -> None:
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