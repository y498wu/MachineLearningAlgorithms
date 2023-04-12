import numpy as np
from typing import Tuple

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