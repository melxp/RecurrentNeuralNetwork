#Three neurons feeding into this neuron, their outputs become the inputs.

inputs = [1, 2, 3]
weights = [0.2, 0.8, -0.5] #every input has a unique weight associated with it
bias = 2 #every unique neuron has a unique bias

output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias

#output = 0
#for i in range(len(inputs)):
    #output += (inputs[i]*weights[i])
#output += bias
