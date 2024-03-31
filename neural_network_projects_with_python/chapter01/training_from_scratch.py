import numpy as np


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)


class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)
        self.layer1 = None
        self.activation = sigmoid
        self.activation_derivative = sigmoid_derivative

    def feedforward(self):
        self.layer1 = self.activation(np.dot(self.input, self.weights1))
        self.output = self.activation(np.dot(self.layer1, self.weights2))

    def backprop(self):
        tmp = 2 * (self.y - self.output) * self.activation_derivative(self.output)
        d_weights2 = np.dot(self.layer1.T, tmp)
        d_weights1 = np.dot(
            self.input.T,
            np.dot(tmp, self.weights2.T) * self.activation_derivative(self.layer1)
        )

        self.weights1 += d_weights1
        self.weights2 += d_weights2


def main():
    nn = NeuralNetwork(
        np.array([
            [0, 0, 1],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ]),
        np.array([
            [0],
            [1],
            [1],
            [0]
        ])
    )

    for i in range(1500):
        nn.feedforward()
        nn.backprop()

    print(nn.output)


if __name__ == '__main__':
    main()
