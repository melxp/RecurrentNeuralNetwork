import numpy as np

# -------------------------------------------------------------------------
# LSTM
# -------------------------------------------------------------------------

class LSTM:
    def __init__(self, n_inputs, n_neurons):

        self.n_neurons = n_neurons

        # One set of weights for: forget gate, input gate, candidate cell state, output gate
        # Every timestep needs four vectors
        self.weights = 0.1 * np.random.randn(n_inputs + n_neurons, 4 * n_neurons)

        self.biases = np.zeros((1, 4 * n_neurons))

        # Store information from every timestep
        self.inputs_history = []
        self.hidden_history = []
        self.cell_history = []

        self.forget_gate_history = []
        self.input_gate_history = []
        self.candidate_history = []
        self.output_gate_history = []

        self.cell_outputs_history = []
        self.outputs_history = []

    def reset_history(self):
        self.inputs_history = []
        self.hidden_history = []
        self.cell_history = []

        self.forget_gate_history = []
        self.input_gate_history = []
        self.candidate_history = []
        self.output_gate_history = []

        self.cell_outputs_history = []
        self.outputs_history = []

    def forward(self, inputs, hidden, cell):

        previous_cell = cell

        # Store inputs and previous states
        self.inputs_history.append(inputs)
        self.hidden_history.append(hidden)
        self.cell_history.append(cell)

        # Combine the current input with the previous hidden state
        combined = np.concatenate((inputs, hidden), axis=1)

        # Calculate all four gate values (shape: (1, 4 * n_neurons))
        gate_values = np.dot(combined, self.weights) + self.biases

        forget_gate = gate_values[:, :self.n_neurons]
        input_gate = gate_values[:, self.n_neurons:2 * self.n_neurons]
        candidate = gate_values[:, 2 * self.n_neurons:3 * self.n_neurons]
        output_gate = gate_values[:, 3 * self.n_neurons:]

        # Apply activation functions to gates
        # Turn raw values -> (forget_gate, input_gate, output_gate -> sigmoid, candidate -> tanh)
        forget_gate = 1 / (1 + np.exp(-forget_gate)) # Decides how much of the old cell memory to keep
        input_gate = 1 / (1 + np.exp(-input_gate)) # Decideas how much new information to write
        candidate = np.tanh(candidate) # New information that could be written into memory
        output_gate = 1 / (1 + np.exp(-output_gate)) # Decides how much of the cell state becomes the new hidden state

        # Store gates
        self.forget_gate_history.append(forget_gate)
        self.input_gate_history.append(input_gate)
        self.candidate_history.append(candidate)
        self.output_gate_history.append(output_gate)


        # Update cell state
        cell = forget_gate * previous_cell + input_gate * candidate

        # Update hidden state
        hidden = output_gate * np.tanh(cell)

        # Store current states
        self.cell_outputs_history.append(cell)
        self.outputs_history.append(hidden)

        return hidden, cell

    def backward(self, dhidden_from_output, dcell_next, timestep):

        #Retrieve saved values
        inputs = self.inputs_history[timestep]
        hidden_previous = self.hidden_history[timestep]
        cell_previous = self.cell_history[timestep]

        forget_gate = self.forget_gate_history[timestep]
        input_gate = self.input_gate_history[timestep]
        candidate = self.candidate_history[timestep]
        output_gate = self.output_gate_history[timestep]

        cell = self.cell_outputs_history[timestep]

        # Backprop through hidden state
        # hidden = output_gate * tanh(cell)
        doutput_gate = dhidden_from_output * np.tanh(cell)

        dcell = dhidden_from_output * output_gate * (1 - np.tanh(cell) ** 2)

        # Add gradient coming from next timestep
        dcell_total = dcell + dcell_next

        # Backprop through cell state
        # cell = forget_gate * cell_previous + input_gate * candidate
        dforget_gate = dcell_total * cell_previous
        dcell_previous = dcell_total * forget_gate
        dinput_gate = dcell_total * candidate
        dcandidate = dcell_total * input_gate

        # Backprop through activation functions
        # Sigmoid derivative
        dforget_raw = dforget_gate * forget_gate * (1 - forget_gate)
        dinput_raw = dinput_gate * input_gate * (1 - input_gate)
        doutput_raw = doutput_gate * output_gate * (1 - output_gate)

        # Tanh derivative
        dcandidate_raw = dcandidate * (1 - candidate ** 2)

        # Combine four gate gradients
        dgate_values = np.concatenate((dforget_raw, dinput_raw, dcandidate_raw, doutput_raw), axis=1)

        # Re-create combined input
        combined = np.concatenate((inputs, hidden_previous), axis=1)

        # Weight gradients
        self.dweights += np.dot(combined.T, dgate_values)
        self.dbiases += dgate_values

        # Gradient into combined input
        dcombined = np.dot(dgate_values, self.weights.T)

        # Split input and hidden gradients
        self.dinputs = dcombined[:, :inputs.shape[1]]
        
        dhidden_previous = dcombined[:, inputs.shape[1]]

        return dhidden_previous, dcell_previous


# -------------------------------------------------------------------------
# OUTPUT LAYER
# -------------------------------------------------------------------------

class OutputLayer:
    def __init__(self, n_inputs, n_outputs):

        self.weights = 0.1 * np.random.randn(n_inputs, n_outputs) 
        self.biases = np.zeros((1, n_outputs))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dhidden_from_output, inputs):

        # Gradient of weights
        self.dweights += np.dot(inputs.T, dhidden_from_output)

        # Gradient of biases
        self.dbiases += dhidden_from_output

        # Gradient passed back to hidden state
        self.dinputs = np.dot(dhidden_from_output, self.weights.T)

    
# -------------------------------------------------------------------------
# TESTING
# -------------------------------------------------------------------------

np.random.seed(42)

lstm = LSTM(n_inputs = 5, n_neurons = 10)

inputs = np.random.randn(1, 5)
hidden = np.zeros((1, 10))
cell = np.zeros((1, 10))

hidden, cell = lstm.forward(inputs, hidden, cell)

print("Hidden shape: ", hidden.shape) # Should be (1, 10)
print("Cell shape: ", cell.shape) # Should be (1, 10)

# Initialise gradients
lstm.dweights = np.zeros_like(lstm.weights)
lstm.dbiases = np.zeros_like(lstm.biases)

# Fake gradient from output
dhidden = np.ones((1, 10))

# No future cell gradient
dcell_next = np.zeros((1, 10))

# Backwards
dhidden_previous, dcell_previous = lstm.backward(dhidden, dcell_next, timestep=0)

print()
print("Backward")
print("dweights shape: ", lstm.dweights.shape)
print("dbiases shape: ", lstm.dbiases.shape)
print("dinputs shape: ", lstm.dinputs.shape)
print("dhidden_previous shape: ", dhidden_previous.shape)
print("dcell_previous shape: ", dcell_previous.shape)