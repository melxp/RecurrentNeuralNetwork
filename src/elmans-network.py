import numpy as np

class RNN:
    def __init__(self, n_inputs, n_neurons):
        self.weights_inputs = 0.1 * np.random.randn(n_inputs, n_neurons) 
        # An Elman RNN needs two sets of weights: current input and previous hidden state
        self.weights_hidden = 0.1 * np.random.randn(n_neurons, n_neurons) 
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs, hidden):
        self.output = np.dot(inputs, self.weights_inputs) + \
                        np.dot(hidden, self.weights_hidden) + \
                        self.biases

class OutputLayer:
    def __init__(self, n_inputs, n_outputs):
        self.weights = 0.1 * np.random.randn(n_inputs, n_outputs) 
        self.biases = np.zeros((1, n_outputs))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

class Loss_CategoricalCrossEntropy:
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        correct_confidence = y_pred_clipped[0, y_true]
        loss = -np.log(correct_confidence)
        return loss



# Create a hard-coded dataset
sentences = [
    "the cat sat",
    "the cat ate",
    "the dog sat",
    "the dog ate"
]

# Get individual words from sentences
vocabulary = set() # Sets don't store repeating values
for sentence in sentences:
    for word in sentence.split():
        vocabulary.add(word)

# Turn words into numbers using a dictionary
word_to_id = {
    "the": 0,
    "cat": 1,
    "sat": 2,
    "ate": 3,
    "dog": 4
}

# Reverse dictionary
id_to_word = {
    0: "the",
    1: "cat",
    2: "sat",
    3: "ate",
    4: "dog"
}

# How this works: 
# When the network makes a prediction, we choose the index with the largest value.
# e.g. [0.05, 0.1, 0.7, 0.1, 0.05] - the largest would be index 2.
# Therefore:
# prediction = np.argmax(output)
# print(id_to_word[prediction])
# would give sat.

# We need to create input to target pairs, which can be automatically generated
inputs = []
targets = []

for sentence in sentences:
    words = sentence.split()

    for i in range(len(words) - 1):
        inputs.append(words[i])
        targets.append(words[i + 1])

# Convert to IDs
inputs = [word_to_id[word] for word in inputs]
targets = [word_to_id[word] for word in targets]

print(inputs)
print(targets)

def one_hot(word_id, vocabulary_size):
    vector = np.zeros(vocabulary_size)
    vector[word_id] = 1
    return vector

rnn = RNN(
    n_inputs = len(word_to_id),
    n_neurons = 10
)

hidden = np.zeros((1, 10))

for input_word in inputs[:2]: # Only using the & cat to keep it simple

    input_vector = one_hot(input_word, len(word_to_id))
    input_vector = input_vector.reshape(1, -1)

    rnn.forward(input_vector, hidden)

    hidden = rnn.output

    print("Input: ", id_to_word[input_word])
    print("Hidden ", hidden)

output_layer = OutputLayer(
    n_inputs = 10,
    n_outputs = len(word_to_id)
)

output_layer.forward(hidden)
print("Output: ", output_layer.output)

activation_softmax = Activation_Softmax()
activation_softmax.forward(output_layer.output)
print("Probabilities", activation_softmax.output)

# Prediction currently random
prediction = np.argmax(activation_softmax.output)
print("Prediction: ", id_to_word[prediction])

loss_function = Loss_CategoricalCrossEntropy()

loss = loss_function.forward(activation_softmax.output, targets[1])

print("Loss: ", loss)