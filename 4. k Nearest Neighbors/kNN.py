import numpy as np
import math
import sys

class KNN:
    """
    K Nearest Neighbours model
    Args:
        k_neigh: Number of neighbours to take for prediction
        weighted: Boolean flag to indicate if the nieghbours contribution
                  is weighted as an inverse of the distance measure
        p: Parameter of Minkowski distance
    """

    def __init__(self, k_neigh, weighted=False, p=2):
        self.weighted = weighted
        self.k_neigh = k_neigh
        self.p = p

    def fit(self, data, target):
        """
        Fit the model to the training dataset.
        Args:
            data: M x D Matrix( M data points with D attributes each)(float)
            target: Vector of length M (Target class for all the data points as int)
        Returns:
            The object itself
        """
        self.data = data
        self.target = target.astype(np.int64)

        return self
    
    def get_minkowski_distance(self, point_1, point_2):
        """
        Calculates the minkowski distance between the two points passed as arguments.
        Args:
            point_1 (numpy array)
            point_2 (numpy array)
        Returns:
            float: minkowski distance between point1 and point2
        """
        minkowski_distance = 0

        for i in range(len(point_1)):
            minkowski_distance += math.pow(abs(point_1[i] - point_2[i]), self.p)
        
        return math.pow(minkowski_distance, 1/(self.p))

    def find_distance(self, x):
        """
        Find the Minkowski distance to all the points in the train dataset x
        Args:
            x: N x D Matrix (N inputs with D attributes each)(float)
        Returns:
            Distance between each input to every data point in the train dataset
            (N x M) Matrix (N Number of inputs, M number of samples in the train dataset)(float)
        """
        n_rows = len(x)
        n_cols = len(self.data)
        
        distance_matrix = np.zeros((n_rows, n_cols))
        
        for i in range(n_rows):
            for j in range(n_cols):
                distance_matrix[i, j] = self.get_minkowski_distance(x[i, :], self.data[j, :])
                
        return distance_matrix

    def k_neighbours(self, x):
        """
        Find K nearest neighbours of each point in train dataset x
        Note that the point itself is not to be included in the set of k Nearest Neighbours
        Args:
            x: N x D Matrix( N inputs with D attributes each)(float)
        Returns:
            k nearest neighbours as a list of (neigh_dists, idx_of_neigh)
            neigh_dists -> N x k Matrix(float) - Dist of all input points to its k closest neighbours.
            idx_of_neigh -> N x k Matrix(int) - The (row index in the dataset) of the k closest neighbours of each input

            Note that each row of both neigh_dists and idx_of_neigh must be SORTED in increasing order of distance
        """
        distance_matrix = self.find_distance(x)
        N = len(x)
        
        neigh_dists = np.zeros((N, self.k_neigh))
        idx_of_neigh = np.zeros((N, self.k_neigh), dtype=np.int64)
        
        for row in range(N):
            distance_list = sorted([(col, distance_matrix[row, col]) for col in range(distance_matrix.shape[1])], key = lambda element: element[1])
            for k in range(self.k_neigh):
                neigh_dists[row, k] = (distance_list[k])[1]
                idx_of_neigh[row, k] = (distance_list[k])[0]
        
        return (neigh_dists, idx_of_neigh)

    def predict(self, x):
        """
        Predict the target value of the inputs.
        Args:
            x: N x D Matrix( N inputs with D attributes each)(float)
        Returns:
            pred: Vector of length N (Predicted target value for each input)(int)
        """
        nPoints = len(x)
        list_outputs = np.zeros(nPoints)
        k_nearest_neighbours = self.k_neighbours(x)
        outputClasses = sorted(np.unique(self.target))
        
        for point in range(nPoints):
            target_dict = dict.fromkeys(outputClasses, 0)
            for dist, idx in zip((k_nearest_neighbours[0])[point, :], (k_nearest_neighbours[1])[point, :]):
                if self.weighted:
                    if dist != 0:
                        target_dict[self.target[idx]] += 1 / dist
                    else:
                        target_dict[self.target[idx]] += sys.maxsize
                else:
                    target_dict[self.target[idx]] += 1
            
            list_outputs[point] = max(target_dict, key = lambda k: target_dict[k])

        return list_outputs

    def evaluate(self, x, y):
        """
        Evaluate Model on test data using 
            classification: accuracy metric
        Args:
            x: Test data (N x D) matrix(float)
            y: True target of test data(int)
        Returns:
            accuracy : (float.)
        """
        n_correct = 0
        y_pred = self.predict(x)
        
        for i in range(len(y)):
            if y_pred[i] == y[i]:
                n_correct += 1
        
        return (n_correct / len(y)) * 100
