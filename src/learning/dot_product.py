import numpy as np

# Dot product of single neuron
inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]    
bias = 2

output = np.dot(weights, inputs) + bias
print(output)

'''
total = 0
for i in range(len(weights)):
    total += weights[i] * inputs[i]
print(total + bias)
'''

# Dot product of layer of neurons
inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

output = np.dot(weights, inputs) + biases
print(output)

'''
output_arr = []
for i in range(len(weights)):
    total = 0
    for j in range(len(weights[0])):
        total += weights[i][j] * inputs[j]
    output_arr.append(total + biases[i])
print(output_arr)
'''

