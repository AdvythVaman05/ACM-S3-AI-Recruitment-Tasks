import numpy as np
import matplotlib.pyplot as plt

class KMeansClustering:
    def __init__(self, k=3, n_init=1):
        self.k = k
        self.n_init = n_init
        self.best_centroids = None
        self.best_inertia = np.inf  # Inertia is the sum of squared distances

    @staticmethod
    def euclidean_distance(data_point, centroids):
        return np.sqrt(np.sum((centroids - data_point) ** 2, axis=1))

    def compute_inertia(self, X, labels, centroids):
        inertia = 0
        for i in range(self.k):
            cluster_points = X[labels == i]
            inertia += np.sum((cluster_points - centroids[i]) ** 2)
        return inertia

    def fit(self, X, max_iterations=40):
        for init_run in range(self.n_init):
            # Initialize centroids randomly within the data range
            centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0), size=(self.k, X.shape[1]))

            for _ in range(max_iterations):
                labels = []

                # Assign clusters based on the nearest centroids
                for data_point in X:
                    distances = KMeansClustering.euclidean_distance(data_point, centroids)
                    cluster_num = np.argmin(distances)
                    labels.append(cluster_num)

                labels = np.array(labels)

                # Update centroids by calculating the mean of points in each cluster
                new_centroids = np.zeros_like(centroids)
                for i in range(self.k):
                    cluster_points = X[labels == i]
                    if len(cluster_points) > 0:
                        new_centroids[i] = np.mean(cluster_points, axis=0)

                # Check for convergence (if centroids don't change much)
                if np.all(np.abs(centroids - new_centroids) < 0.0001):
                    break

                centroids = new_centroids

            # Calculate inertia for the current run
            inertia = self.compute_inertia(X, labels, centroids)

            # If this run has the best result so far, save the centroids and inertia
            if inertia < self.best_inertia:
                self.best_inertia = inertia
                self.best_centroids = centroids

        # Return the labels from the best run
        return labels

    def predict(self, X):
        labels = []
        for data_point in X:
            distances = KMeansClustering.euclidean_distance(data_point, self.best_centroids)
            labels.append(np.argmin(distances))
        return np.array(labels)
