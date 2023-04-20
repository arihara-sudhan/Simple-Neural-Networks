import numpy as np

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)

    def forward(self, inputs):
        self.hidden = sigmoid(np.dot(inputs, self.weights1))
        self.output = sigmoid(np.dot(self.hidden, self.weights2))
        return self.output

    def backward(self, inputs, outputs, learning_rate):
        # Perform the backward pass
        output_error = outputs - self.output
        output_delta = output_error * sigmoid(self.output) * (1 - sigmoid(self.output))
        hidden_error = np.dot(output_delta, self.weights2.T)
        hidden_delta = hidden_error * sigmoid(self.hidden) * (1 - sigmoid(self.hidden))
        self.weights2 += learning_rate * np.dot(self.hidden.T, output_delta)
        self.weights1 += learning_rate * np.dot(inputs.T, hidden_delta)

# Define the input, hidden, and output sizes
input_size = 2
hidden_size = 3
output_size = 1

# Create a neural network with the given sizes
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Train the neural network using a training set
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[0], [1], [1], [1]])

for i in range(10000):
    nn.forward(inputs)
    nn.backward(inputs, outputs, 0.1)

test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
test_outputs = nn.forward(test_inputs)
print(test_outputs)
