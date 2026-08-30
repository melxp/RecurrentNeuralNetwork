import numpy as np
import nnfs # For the tutorial
from nnfs.datasets import spiral_data # Import spiral dataset

nnfs.init() # For the tutorial
#np.random.seed(0)

# First layer from input data (X)
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons) # Opposite way around so we don't need to transpose everytime we do a forward pass
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# Activation function using Rectified Linear Unit (ReLU)
# Takes all values from neurons and produces the activation for the entire layer
# You can have different activation functions in a layer
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

# Activation function using Softmax
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # Prevent overflow
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

# Output layer
dense2 = Layer_Dense(3, 3) # 3 neurons for output because 3 classes
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5]) # Only printing first 5, as ther should be 300