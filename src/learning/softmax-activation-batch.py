import numpy as np
import nnfs # For tutorial

nnfs.init() # For tutorial

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_outputs) # Get exponentials
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True) # Normalize

print(norm_values)
print(sum(norm_values)) 
