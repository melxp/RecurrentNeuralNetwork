# Using objects

import numpy as np

np.random.seed(0)

# Input data to neural network
X = [[1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) # Opposite way around so we don't need to transpose everytime we do a forward pass
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = Layer_Dense(4, 5) # Output from layer1 is the input to layer2
layer2 = Layer_Dense(5, 2)

layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)