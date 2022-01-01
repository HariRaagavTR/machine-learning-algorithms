from random import random
import csv

class Perceptron:
    def __init__(self, n_inputs):
        self.n = n_inputs
        self.weights = [random() for i in range(n_inputs + 1)]

    # Step Function
    def activation_function(self, net_i):
        if net_i >= 0:
            return 1
        else:
            return 0

    # Summation Function
    def sigma_function(self, percept):
        net_i = 0
        for _input, _weight in zip([1] + percept, self.weights):
            net_i += _input * _weight
        return net_i

    # Output Function
    def output(self, percept):
        return self.activation_function(self.sigma_function(percept))

    # Error Percentage
    def error(self, inputs, outputs):
        n_incorrect = 0
        for data_point, actual_output in zip(inputs, outputs):
            if self.output(data_point) != actual_output:
                n_incorrect += 1
        return (n_incorrect / len(inputs)) * 100

    # Perceptron Learning Algorithm
    def learn(self, X, Y, learning_rate = 0.1, error_threshold = 0.0, max_epochs = 100, print_message = True):
        input_length = len(X)
        it_idx = 0
        n_epochs = 1
        while True:
            inputs = X[it_idx]
            output = Y[it_idx]
            predicted_output = self.output(inputs)
            
            if output != predicted_output:
                for idx in range(len(self.weights)):
                    self.weights[idx] += learning_rate * (output - predicted_output) * ([1] + inputs)[idx]
            
            it_idx += 1
            if it_idx == input_length:
                error = self.error(X, Y)
                print('Epoch #', n_epochs, ' Complete. Accuracy: ', 100.0 - error, '%.', sep = '')
                n_epochs += 1
                if error <= error_threshold or n_epochs > max_epochs:
                    break
                else:
                    it_idx = 0


# Training the Perceptron.
file_name = 'iris.csv'
X = []
Y = []
with open(file_name, 'r') as file:
    data = csv.reader(file, delimiter = ',')
    for row in data:
        X.append([float(element) for element in row[:-1]])
        Y.append(int(row[-1]))
n_inputs = len(X[0])

perceptron = Perceptron(n_inputs)
perceptron.learn(X, Y, learning_rate = 0.01)

# Testing (To Be Added)
