import math

layer_outputs = [4.8, 1.21, 2.385] # Intended target = largest

#E = 2.71828182846
E = math.e

exp_values = []

for output in layer_outputs:
    exp_values.append(E**output) # Convert negatives to positives without losing the meaning of a negative value
print(exp_values) 

norm_base = sum(exp_values)
norm_values = []

# Normalise
for value in exp_values:
    norm_values.append(value / norm_base)

print(norm_values)
print(sum(norm_values)) # Should be 1 or very very close to 1

# NumPy version:
'''
import numpy as np

layer_outputs = [4.8, 1.21, 2.385] 

exp_values = np.exp(layer_outputs) # Get exponentials

norm_values = exp_values / np.sum(exp_values) # Normalize

print(norm_values)
print(sum(norm_values)) 
'''