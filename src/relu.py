import numpy as np
import nnfs # For the tutorial
from nnfs.datasets import spiral_data # Import spiral dataset

nnfs.init() # For the tutorial
#np.random.seed(0)

# Input data to neural network
X = [[1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]

X, y = spiral_data(100, 3) # 100 feature sets, 3 classes

# First layer from input data (X)
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) # Opposite way around so we don't need to transpose everytime we do a forward pass
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# Activation function using Rectified Linear Unit 
# Takes all values from neurons and produces the activation for the entire layer
# You can have different activation functions in a layer
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


layer1 = Layer_Dense(2, 5) # No. inputs, No. neurons (2 features in dataset)
activation1 = Activation_ReLU()

layer1.forward(X)

#print(layer1.output)
activation1.forward(layer1.output) # All negatives turn to 0 with ReLU
print(activation1.output)