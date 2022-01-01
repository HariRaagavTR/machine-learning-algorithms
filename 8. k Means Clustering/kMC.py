import numpy as np
import math

class KMeansClustering:
    """
    K-Means Clustering Model

    Args:
        n_clusters: Number of clusters(int)
    """
    def __init__(self, n_clusters, n_init=10, max_iter=1000, delta=0.001):

        self.n_cluster = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.delta = delta

    def init_centroids(self, data):
        idx = np.random.choice(
            data.shape[0], size=self.n_cluster, replace=False)
        self.centroids = np.copy(data[idx, :])

    def fit(self, data):
        """
        Fit the model to the training dataset.
        Args:
            data: M x D Matrix(M data points with D attributes each)(numpy float)
        Returns:
            The object itself
        """
        if data.shape[0] < self.n_cluster:
            raise ValueError(
                'Number of clusters is grater than number of datapoints')

        best_centroids = None
        m_score = float('inf')

        for _ in range(self.n_init):
            self.init_centroids(data)

            for _ in range(self.max_iter):
                cluster_assign = self.e_step(data)
                old_centroid = np.copy(self.centroids)
                self.m_step(data, cluster_assign)

                if np.abs(old_centroid - self.centroids).sum() < self.delta:
                    break

            cur_score = self.evaluate(data)

            if cur_score < m_score:
                m_score = cur_score
                best_centroids = np.copy(self.centroids)

        self.centroids = best_centroids

        return self

    def e_step(self, data):
        """
        Expectation Step.
        Finding the cluster assignments of all the points in the data passed
        based on the current centroids
        Args:
            data: M x D Matrix (M training samples with D attributes each)(numpy float)
        Returns:
            Cluster assignment of all the samples in the training data
            (M) Vector (M number of samples in the train dataset)(numpy int)
        """
        M, D = data.shape
        distanceMatrix = np.zeros((self.n_cluster, M))

        for idxCentroid, centroid in enumerate(self.centroids):
            for idxPoint, point in enumerate(data):
                distanceMatrix[idxCentroid, idxPoint] = np.linalg.norm(point - centroid)
        
        return np.argmin(distanceMatrix, axis = 0)


    def m_step(self, data, cluster_assgn):
        """
        Maximization Step.
        Compute the centroids.
        Args:
            data: M x D Matrix (M training samples with D attributes each) (numpy float)
            Change self.centroids
        """
        M, D = data.shape
        newCentroids = np.zeros((self.n_cluster, D))
        newCluterSizes = np.zeros(self.n_cluster)

        for point, cluster in zip(data, cluster_assgn):
            newCluterSizes[cluster] += 1
            newCentroids[cluster] += point
        
        for cluster in range(self.n_cluster):
            if newCluterSizes[cluster] != 0:
                newCentroids[cluster] /= newCluterSizes[cluster]
        
        self.centroids = newCentroids

    def evaluate(self, data, cluster_assign):
        """
        K-Means Objective
        Args:
            data: Test data (M x D) matrix (numpy float)
            cluster_assign: M vector, Cluster assignment of all the samples in `data`
        Returns:
            metric : (float.)
        """
        sumOfSquaresError = 0
        
        for point, cluster in zip(data, cluster_assign):
            sumOfSquaresError += math.pow(np.linalg.norm(point - self.centroids[cluster]), 2)
        
        return sumOfSquaresError
