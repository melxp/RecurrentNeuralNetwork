import numpy as np

np.random.seed(42)

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
        self.outputs_history = []

    def reset_history(self):
        self.inputs_history = []
        self.hidden_history = []
        self.cell_history = []
        self.outputs_history = []

    def forward(self, inputs, hidden, cell):

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

        # Update cell state
        cell = forget_gate * cell + input_gate * candidate

        # Update hidden state
        hidden = output_gate * np.tanh(cell)

        # Store current states
        self.cell_history.append(cell)
        self.outputs_history.append(hidden)

        return hidden, cell
