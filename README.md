# Elman Recurrent Neural Network 

I'm currently building an Elman Recurrent Neural Network from scratch in Python and NumPy to improve my understanding of how recurrent neural networks work and how the underlying mathematics can be translated into code.

I'm building the network without using a machine learning framework, implementing the main components myself and experimenting with different datasets and parameters to see how they affect what the network learns.

## What I've Implemented

So far, the network includes:

- Word tokenisation and vocabulary creation
- Word-to-ID and ID-to-word conversion
- One-hot encoding
- An Elman recurrent hidden layer
- `tanh` activation
- An output layer
- Softmax activation
- Categorical cross-entropy loss
- Forward propagation
- Backpropagation Through Time (BPTT)
- Gradient descent
- Gradient clipping
- Loss tracking
- Word prediction
- Temperature-based text generation

## How It Works

The network takes a sequence of words as input and predicts the next word in the sequence. 

Each word is first converted into a one-hot encoded vector. This is passed into the recurrent layer along with the hidden state from the previous timestep. 

The hidden state is calculated using the current input, the previous hidden state, and the network's weights. A `tanh` activation is then applied before passing the result to the output layer. 

The output is converted into probabilities using softmax, and categorical cross-entropy is used to calculate the loss. 

During training, the error is propagated backwards through the sequence using Backpropagation Through Time. The gradients are then used to update the weights using gradient descent. 

Gradient clipping has also been added to help prevent exploding gradients during training.

## Dataset

I'm currently using a small, hard-coded dataset of simple sentences to experiment with how the network learns patterns in sequences. 

I started with shorter and simpler sequences, and I am slowly adding more dependencies to make the sequences more complex.

This allows me to experiment with how the network responds as the amount of information it needs to learn increases.

## Results

The following results are from the current dataset and model:

```text
Epoch:  0
Average Loss:  3.184795121037979

Epoch:  100
Average Loss:  2.9344489992359453

Epoch:  200
Average Loss:  2.045770168374786

Epoch:  300
Average Loss:  1.4008619037042727

Epoch:  400
Average Loss:  1.0066182033267366

Epoch:  500
Average Loss:  0.7637801722307962

Epoch:  600
Average Loss:  0.6256954198149256

Epoch:  700
Average Loss:  0.5412983585842256

Epoch:  800
Average Loss:  0.48547333053644925

Epoch:  900
Average Loss:  0.44333603636697544
```

The loss decreases as training progresses, which suggests that the network is learning patterns from the current dataset.

## Text Generation

Once trained, the network can generate text by feeding its predictions back into the network. I have also implemented temperature sampling to alter how random the generated text is. The lower the temperature, the more predictable the output.

```text
Temperature: 0.2

Input: the
Generated: the small rabbit that was quiet and very playful sat on the mat
```

## What I'm Learning From

I wanted to build this project because we learnt about Elman's Recurrent Neural Network during my first year of university, and it really piqued my interest. Since then, I have been learning through DeepLearning.AI's Machine Learning Specialization course, as well as working through Sentdex's tutorial on building a recurrent neural network from scratch in Python. 

I've also been using additional documentation, explanations and resources to help me understand the concepts and work through problems as I build the network.

I'm using these resources as a starting point and a helping hand, but I am implementing and experimenting with the network myself to improve my understanding of how the different components work together.

## Technologies
- Python
- NumPy
- Machine Learning
- Recurrent Neural Networks
- Backpropagation Through Time
- Gradient Descent
