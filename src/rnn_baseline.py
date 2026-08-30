import numpy as np

np.random.seed(42)

# -------------------------------------------------------------------------
# RNN
# -------------------------------------------------------------------------

class RNN:
    def __init__(self, n_inputs, n_neurons):

        # Inputs to hidden weights
        self.weights_inputs = 0.1 * np.random.randn(n_inputs, n_neurons) 

        # Weight connecting previous hidden state to current
        self.weights_hidden = 0.1 * np.random.randn(n_neurons, n_neurons) 

        self.biases = np.zeros((1, n_neurons))

        # Store information from every timestep
        self.inputs_history = []
        self.hidden_history = []
        self.outputs_history = []

    def reset_history(self):
        self.inputs_history = []
        self.hidden_history = []
        self.outputs_history = []

    def forward(self, inputs, hidden):

        # Store inputs and previous hidden state
        self.inputs_history.append(inputs)
        self.hidden_history.append(hidden)

        # RNN equation to calculate hidden state
        self.output = np.dot(inputs, self.weights_inputs) + \
                        np.dot(hidden, self.weights_hidden) + \
                        self.biases

        # Add non-linearity
        self.output = np.tanh(self.output)

        # Store current hidden state
        self.outputs_history.append(self.output)

        return self.output

    def backward(self, dhidden_from_output, timestep, dhidden_next):

        # Get values from timestep
        inputs = self.inputs_history[timestep]
        hidden = self.hidden_history[timestep]
        output = self.outputs_history[timestep]

        # Add gradient coming from next timestep
        # Gradient at this timestep = gradient from output + gradient from future hidden state
        dhidden_total = dhidden_from_output + dhidden_next

        # Backprop through tanh
        dtanh = dhidden_total * (1 - output ** 2)

        # Calculate gradients for the RNN weights
        # Added because the same weights are used at every timestep
        self.dweights_inputs += np.dot(inputs.T, dtanh)
        self.dweights_hidden += np.dot(hidden.T, dtanh)

        # Gradient of biases
        self.dbiases += dtanh

        # Gradient passed backwards into the input
        self.dinputs = np.dot(dtanh, self.weights_inputs.T)

        # Gradient passed backwards into the previous hidden state
        self.dhidden = np.dot(dtanh, self.weights_hidden.T)


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
# SOFTMAX
# -------------------------------------------------------------------------

class Activation_Softmax:
    def forward(self, inputs):

        # Subtract max for stability
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

# -------------------------------------------------------------------------
# CATEGORICAL CROSS-ENTROPY LOSS
# -------------------------------------------------------------------------

class Loss_CategoricalCrossEntropy:
    def forward(self, y_pred, y_true):

        # Prevent log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probability assigned to correct class
        correct_confidence = y_pred_clipped[0, y_true]

        # Cross-entropy loss
        loss = -np.log(correct_confidence)

        return loss

# -------------------------------------------------------------------------
# DATASET
# -------------------------------------------------------------------------

# Create a hard-coded dataset with <END> tokens

'''
# Dataset 1: Short, simple sequences
# ~0.95 loss
sentences = [
    "the cat sat <END>",
    "the cat ate <END>",
    "the cat ran <END>",
    "the cat slept <END>",
    "the dog sat <END>",
    "the dog ate <END>",
    "the dog ran <END>",
    "the dog slept <END>"
    "the rabbit sat <END>",
    "the rabbit ate <END>",
    "the rabbit ran <END>",
    "the rabbit slept <END>"
]
'''

'''
# Dataset 2: Longer sequences
# ~1.1 loss
sentences = [
    "the cat sat on the mat <END>",
    "the cat caught the fish <END>",
    "the cat licked the toy <END>",
    "the dog ate from the bowl <END>",
    "the dog ran to the park <END>",
    "the dog ate the chicken <END>",
    "the rabbit bit the carrot <END>",
    "the rabbit ate the carrot <END>",
    "the rabbit sat on the mat <END>"
]
'''

'''
# Dataset 3: Combinations (sat -> on/by, ate -> the/from)
# ~0.86 loss
sentences = [
    "the cat sat on the mat <END>",
    "the cat ate the fish <END>",
    "the dog sat by the bed <END>",
    "the dog ate from the bowl <END>",
    "the rabbit sat on the grass <END>",
    "the rabbit ate the carrot <END>"
]
'''

# Dataset 4: Long range dependencies
# Vanilla RNN baseline: ~0.44 loss
sentences = [
    "the small cat that was quiet and very friendly sat on the mat <END>",
    "the big dog that was loud and very playful sat on the bed <END",
    
    "the small rabbit that was quiet and very gentle ate the carrot <END>",
    "the big rabbit that was loud and very energetic ate the carrot <END>", 
]


# -------------------------------------------------------------------------
# VOCABULARY
# -------------------------------------------------------------------------

# Get individual words from sentences
vocabulary = set() # Sets don't store repeating values
for sentence in sentences:
    for word in sentence.split():
        vocabulary.add(word)

# Sort vocabulary for consistent IDs
vocabulary = sorted(vocabulary)

# -------------------------------------------------------------------------
# WORD => ID
# -------------------------------------------------------------------------

# Turn words into numbers using a dictionary
word_to_id = {
    word: i
    for i, word in enumerate(vocabulary)
}

# -------------------------------------------------------------------------
# ID -> WORD
# -------------------------------------------------------------------------

# Reverse dictionary
id_to_word = {
    i: word
    for word, i in word_to_id.items()
}

# -------------------------------------------------------------------------

# How this works: 
# When the network makes a prediction, we choose the index with the largest value.
# e.g. [0.05, 0.1, 0.7, 0.1, 0.05] - the largest would be index 2.
# Therefore:
# prediction = np.argmax(output)
# print(id_to_word[prediction])
# would give sat.

# -------------------------------------------------------------------------
# SEQUENCES
# -------------------------------------------------------------------------

sequences = []

for sentence in sentences:
    words = sentence.split()
    word_ids = [word_to_id[word] for word in words]
    sequences.append(word_ids)
    
print("Sequences: ", sequences)


# -------------------------------------------------------------------------
# CREATE INPUT / TARGET PAIRS
# -------------------------------------------------------------------------

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

# -------------------------------------------------------------------------
# ONE-HOT ENCODING
# -------------------------------------------------------------------------

def one_hot(word_id, vocabulary_size):
    vector = np.zeros(vocabulary_size)
    vector[word_id] = 1
    return vector

# -------------------------------------------------------------------------
# CREATE NETWORK
# -------------------------------------------------------------------------

rnn = RNN(
    n_inputs = len(word_to_id),
    n_neurons = 10
)

output_layer = OutputLayer(
    n_inputs = 10,
    n_outputs = len(word_to_id)
)

activation_softmax = Activation_Softmax()

loss_function = Loss_CategoricalCrossEntropy()

# -------------------------------------------------------------------------
# TRAINING SETTINGS
# -------------------------------------------------------------------------

learning_rate = 0.01
epochs = 1000 # Loss should decrease from epoch 0 towards epoch 1000

# -------------------------------------------------------------------------
# TRAINING LOOP
# -------------------------------------------------------------------------

for epoch in range(epochs):

    total_loss = 0

    # Reset gradients once per epoch
    output_layer.dweights = np.zeros_like(output_layer.weights)
    output_layer.dbiases = np.zeros_like(output_layer.biases)

    # Reset RNN gradients
    rnn.dweights_inputs = np.zeros_like(rnn.weights_inputs)
    rnn.dweights_hidden = np.zeros_like(rnn.weights_hidden)
    rnn.dbiases = np.zeros_like(rnn.biases)

    # Loop through every sentence
    for sequence in sequences:

        # Initial hidden state (0)
        hidden = np.zeros((1, 10))

        # Reset RNN memory
        rnn.reset_history()

        # Store gradient from output layer to RNN for every timestep
        output_gradients = []

        # -------------------------------------------------------------------------
        # FORWARD PASS
        # -------------------------------------------------------------------------

        for timestep in range(len(sequence) - 1):

            input_word = sequence[timestep]
            target_word = sequence[timestep + 1]

            # Convert input word to one-hot vector
            input_vector = one_hot(input_word, len(word_to_id)).reshape(1, -1)

            # RNN forward
            hidden = rnn.forward(input_vector, hidden)

            # Output layer forward
            output_layer.forward(hidden)

            # SoftMax
            activation_softmax.forward(output_layer.output)

            # Loss
            loss = loss_function.forward(activation_softmax.output, target_word)

            total_loss += loss

            # SoftMax & Cross-Entropy gradient
            dhidden_from_output = activation_softmax.output.copy()
            dhidden_from_output[0, target_word] -= 1

            # Output layer backward
            output_layer.backward(dhidden_from_output, hidden)

            # Save gradient from timestep
            output_gradients.append(output_layer.dinputs.copy())  

        # -------------------------------------------------------------------------
        # BACKPROP
        # -------------------------------------------------------------------------

        # No next gradient at final timestep
        dhidden_next = np.zeros((1, 10))

        # Move backwards through sequence
        for timestep in reversed(range(len(sequence) - 1)):
            rnn.backward(output_gradients[timestep], timestep, dhidden_next)

            # Pass gradient to previous timestep
            dhidden_next = rnn.dhidden

    # -------------------------------------------------------------------------
    # CLIP AND UPDATE WEIGHTS
    # -------------------------------------------------------------------------
            
    # Average gradients across all sequences
    num_sequences = len(sequences)

    output_layer.dweights /= num_sequences
    output_layer.dbiases /= num_sequences

    rnn.dweights_inputs /= num_sequences
    rnn.dweights_hidden /= num_sequences
    rnn.dbiases /= num_sequences

    # Clip gradients before updating to prevent exploding gradients
    np.clip(output_layer.dweights, -5, 5, out=output_layer.dweights)
    np.clip(output_layer.dbiases, -5, 5, out=output_layer.dbiases)

    np.clip(rnn.dweights_inputs, -5, 5, out=rnn.dweights_inputs)
    np.clip(rnn.dweights_hidden, -5, 5, out=rnn.dweights_hidden)
    np.clip(rnn.dbiases, -5, 5, out=rnn.dbiases)

    # Update output layer weights
    output_layer.weights -= learning_rate * output_layer.dweights
    output_layer.biases -= learning_rate * output_layer.dbiases

    # Update RNN weights
    rnn.weights_inputs -= learning_rate * rnn.dweights_inputs
    rnn.weights_hidden -= learning_rate * rnn.dweights_hidden
    rnn.biases -= learning_rate * rnn.dbiases

    # -------------------------------------------------------------------------
    # PRINT LOSS
    # -------------------------------------------------------------------------

    total_predictions = sum(len(sequence) - 1 for sequence in sequences)
    average_loss = total_loss / total_predictions

    if epoch % 100 == 0:
        print("Epoch: ", epoch)
        print("Average Loss: ", average_loss)
    

# -------------------------------------------------------------------------
# TESTING
# -------------------------------------------------------------------------

print()
print("==========")
print("TRAINING COMPLETE")
print("==========")

test_sequence = sequences[0]

hidden = np.zeros((1, 10))
rnn.reset_history()

for timestep in range(len(test_sequence) - 1):

    input_word = test_sequence[timestep]
    target_word = test_sequence[timestep + 1]

    input_vector = one_hot(input_word, len(word_to_id)).reshape(1, -1)

    rnn.forward(input_vector, hidden)
    hidden = rnn.output

    output_layer.forward(hidden)
    activation_softmax.forward(output_layer.output)

    prediction = np.argmax(activation_softmax.output)

    # Print results
    print()
    print("Input: ", id_to_word[input_word])
    print("Target: ", id_to_word[target_word])
    print("Prediction: ", id_to_word[prediction])
    print("Probabilities: ", activation_softmax.output)


# -------------------------------------------------------------------------
# TEXT GENERATION
# -------------------------------------------------------------------------


def sample_prediction(probabilities, temperature = 1.0):

    probabilities = np.asarray(probabilities, dtype=np.float64)

    # Temperature controls randomness
    probabilities = np.log(probabilities + 1e-7) / temperature

    probabilities = np.exp(probabilities - np.max(probabilities))
    probabilities /= np.sum(probabilities)

    prediction = np.random.choice(len(probabilities), p=probabilities)

    return prediction

def generate_text(start_word, number_of_words, temperature = 1.0):


    current_word = word_to_id[start_word]
    hidden = np.zeros((1, 10))
    rnn.reset_history()

    generated_words = [start_word]

    # Generate one word at a time
    for _ in range(number_of_words):
        input_vector = one_hot(current_word, len(word_to_id)).reshape(1, -1)

        rnn.forward(input_vector, hidden)
        hidden = rnn.output

        output_layer.forward(hidden)

        activation_softmax.forward(output_layer.output)

        # Choose most likely word
        #prediction = np.argmax(activation_softmax.output)

        # Sampling from probability distribution
        # To give other words a chance, instead of always choosing the max
        prediction = sample_prediction(activation_softmax.output[0], temperature)

        predicted_word = id_to_word[prediction]

        # Stop if the model predicts <END> token
        if predicted_word == "<END>":
            break

        generated_words.append(predicted_word)

        # Feed prediction back into RNN
        current_word = prediction

    return " ".join(generated_words)

print()
print("==========")
print("GENERATED TEXT")
print("==========")

print(generate_text("the", 20, temperature = 0.2))
print(generate_text("the", 20, temperature = 0.5))
print(generate_text("cat", 20, temperature = 0.2))
print(generate_text("dog", 20, temperature = 0.2))

# Temperature examples: 
# 0.2 - very predictable
# 1.0 - normal sampling
# 2.0 - more random