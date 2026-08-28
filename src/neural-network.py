import numpy as np

class NeuralNetwork:
    def __init__(self, input_size=784, hidden_layers=[512, 512], output_size=10):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.weights = []
        self.biases = []
        self.gradientWeights = []
        self.gradientBiases = []
        self.iterations = 0

        # Input to hidden layers network
        np.random.seed(0)
        self.weights.append(0.01 * np.random.randn(input_size, hidden_layers[i+1]))
        self.biases.append(np.zeros((1, hidden_layers[0])))

        # Hidden layers network
        for i in range(len(hidden_layers)-1):
            np.random.seed(0)
            self.weights.append(0.01 * np.random.randn(hidden_layers[i], hidden_layers[i+1]))
            self.biases.append(np.zeros((1, hidden_layers[i + 1])))

        # Hidden layers network to output
        np.random.seed(0)
        self.weights.append(0.01 * np.random.randn(hidden_layers[len(hidden_layers)-1], output_size)) 
        self.biases.append(np.zeros((1, output_size)))   

    def forward(self, inputs):
        self.outputs = [inputs]
        self.outputsTesting = ["inputs"]

        for i in range(len(self.weights)):
            # Dot product to
            self.outputs.append(np.dot(self.outputs[-1], self.weights[i]) + self.biases[i])
            self.outputsTesting.append("dense")

            #Activation functions (ReLU & SoftMax)
            if i == len(self.weights)-1:
                finalOutput = np.exp(self.outputs[-1] - np.max(self.outputs[-1], axis=1, keepdims=True))
                finalOutput = finalOutput / np.sum(finalOutput, axis=1, keepdims=True)
                self.outputs.append(finalOutput)
                self.outputsTesting.append("softmax")
            else:
                self.outputs.append(np.maximum(0, self.outputs[-1]))
                self.outputsTesting.append("relu")

        return self.outputs[-1]

    def backwards(self, y_true):
        # Softmax & LossCategoricalEntropy

        #Number of samples
        samples = len(self.outputs[-1])

        # If labels are one-hot encoded, turn them into discrete values
        if len(y_true.shape) == 2:
            print("Changing to Discrete Values")
            y_true = np.argmax(y_true, axis=1)

        # Copy to safely modify
        dSoftMaxCrossEntropy = self.outputs[-1].copy()
        # Calculate gradient
        dSoftMaxCrossEntropy[range(samples), y_true] -= 1
        # Normalize gradient
        dSoftMaxCrossEntropy = dSoftMaxCrossEntropy / samples

        # Calculate gradients -> calculate derivatives of weights, biases, and inputs 
        dInputs = np.dot(dSoftMaxCrossEntropy.copy(), self.weights[-1].T)

        dWeights = np.dot(self.outputs[-3].T, dSoftMaxCrossEntropy.copy())
        dBiases = np.sum(dSoftMaxCrossEntropy.copy(), axis=0, keepdims=True)
        self.gradientWeights = [dWeights] + self.gradientWeights
        self.gradientBiases = [dBiases] + self.gradientBiases

        i = -3
        j = -1
        for _ in range(len(self.hidden_layers)):
            i -= 1
            j -= 1

            # ReLU activation function
            dInputsReLU = dInputs.copy()
            dInputsReLU[self.outputs[i] <= 0] = 0

            i -= 1
            dInputs = np.dot(dInputsReLU, self.weights[j].T)
            dWeights = np.dot(self.outputs[i].T, dInputsReLU)
            dBiases = np.sum(dInputsReLU, axis=0, keepdims=True)
            self.gradientWeights = [dWeights] + self.gradientWeights
            self.gradientBiases = [dBiases] + self.gradientBiases

        '''
        print("dense1.dweights")
        print(dWeights)
        print("dWeights.shape)
        print("dense1.dbiases")
        print(dBiases)
        print(dBiases.shape)
        '''

    def updateParams(self, lr=0.05, decay=1e-7):
        lr = lr * (1. / (1. + decay * self.iterations))
        # print(f"Learning Rate: {lr}")
        # print(self.iterations)

        for i in range(len(self.weights)-1):
            assert self.weights[i].shape == self.gradientWeights[i].shape
            self.weights[i] += -lr*self.gradientWeights[i]

        for i in range(len(self.biases)-1):
            assert self.biases[i].shape == self.gradientsBiases[i].shape
            self.biases[i] += -lr*self.gradientsBiases[i]

        # print(f"Learning Rate: {lr}")

        self.iterations += 1

    def LossCategoricalCrossEntropy(yPred, yTrue):
        # If predicted class has a prediction of 0%, likelihood this prevents log(0), which would be infinity
        yPred = np.clip(yPred, 1e-10, 1 - 1e-10)

        # Calculate sum of log losses
        loss = -np.sum(yTrue * np.log(yPred), axis=1)

        # Calculate the average loss
        avg_loss = np.mean(loss)

        return avg_loss    
