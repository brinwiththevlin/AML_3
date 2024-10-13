import numpy as np


class Graph:
    def __init__(self, K: np.ndarray):
        self.W = self.similairyGraph(K)

    def similairyGraph(self, K: np.ndarray, threshold: float = 0.5):
        """
        Create a similarity graph from a kernel matrix
        """
        distances = K.copy()
        d_max = distances.max()
        distances = d_max - distances

        distances[distances < threshold * d_max] = 0
        return distances


def spectralClustering(W: np.ndarray, k: int):
    """
    Spectral Clustering
    """

    sim_graph = Graph(W)
    W = sim_graph.W
    L = np.diag(W.sum(axis=1)) - W

    # compute the eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(L)
    U = eigvecs[:, :k]
    clusters = kmeans(U, k)
    return clusters


def kmeans(X: np.ndarray, k: int):
    """
    K-means
    """

    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    prev_centroids = np.zeros(centroids.shape)
    clusters = np.zeros(X.shape[0])
    error = np.linalg.norm(centroids - prev_centroids)
    while error != 0:
        for i in range(X.shape[0]):
            distances = np.linalg.norm(X[i] - centroids, axis=1)
            clusters[i] = np.argmin(distances)
        prev_centroids = centroids
        for i in range(k):
            centroids[i] = np.mean(X[clusters == i], axis=0)
        error = np.linalg.norm(centroids - prev_centroids)
    return clusters
