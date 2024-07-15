import numpy as np


class FullyConnectedLayer:
    def __init__(self, inputs, outputs, activation='sigmoid', optimizer='sgd',
                    learning_rate=0.1, beta=0.9, epsilon=1e-8, m_beta=0.9, v_beta=0.999):
        self.weights = np.random.standard_normal((inputs, outputs))
        self.input = None
        self.output = None
        self.delta = None
        self.bias = None
        self.momentum = np.zeros((inputs, outputs))
        self.beta = beta
        self.epsilon = epsilon
        self.m_average = np.zeros((inputs, outputs))
        self.v_average = np.zeros((inputs, outputs))
        self.m_beta = m_beta
        self.v_beta = v_beta
        self.activation = activation
        self.optimizer = optimizer
        self.learning_rate = learning_rate

    def forward(self, inputs):
        """
            Forward the input to the layer
        """
        self.input = inputs
        self.bias = 0.1 * np.ones((self.input.shape[0], self.weights.shape[1]))
        if self.activation == 'none':
            self.output = np.dot(self.input, self.weights) + self.bias
        else:
            self.output = self.getActivation(np.dot(self.input, self.weights) + self.bias)
        return self.output

    def backward(self, loss):
        """
            Backward the loss to the layer
            - Update the delta of the layer
            - Return the loss to the previous layer
        """
        if self.activation == 'none':
            self.delta = loss
        else:
            self.delta = np.multiply(self.getDerivativeActivation(self.output), loss)
        return np.dot(self.delta, self.weights.T)

    def optimization(self):
        """
            Update the weights of the layer
        """
        gradient = np.dot(self.input.T, self.delta)
        if self.optimizer == 'sgd':
            self.sgd(gradient)
        elif self.optimizer == 'momentum':
            self.momentum(gradient)
        elif self.optimizer == 'adagrad':
            self.adagrad(gradient)
        elif self.optimizer == 'adam':
            self.adam(gradient)

    def sgd(self, gradient):
        self.weights -= self.learning_rate * gradient

    def momentum(self, gradient):
        self.momentum = self.beta * self.momentum - self.learning_rate * gradient
        self.weights += self.momentum

    def adagrad(self, gradient):
        gradient_square = np.sum(gradient ** 2)
        self.weights -= self.learning_rate * (1 / np.sqrt(gradient_square + self.epsilon)) * gradient

    def adam(self, gradient):
        self.m_average = self.m_beta * self.m_average + (1 - self.m_beta) * gradient
        self.v_average = self.v_beta * self.v_average + (1 - self.v_beta) * (gradient ** 2)
        m_bias = self.m_average / (1 - self.m_beta)
        v_bias = self.v_average / (1 - self.v_beta)
        self.weights -= self.learning_rate * m_bias / np.sqrt(v_bias) + self.epsilon

    def getActivation(self, x):
        if self.activation == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-x))
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return np.maximum(0.0, x)
        elif self.activation == 'leaky_relu':
            return np.maximum(0.0, x) + 0.01 * np.minimum(0.0, x)

    def getDerivativeActivation(self, x):
        if self.activation == 'sigmoid':
            return np.multiply(x, 1.0 - x)
        elif self.activation == 'tanh':
            return 1 - x ** 2
        elif self.activation == 'relu':
            return np.greater(x, 0).astype(int)
        elif self.activation == 'leaky_relu':
            return np.greater(x, 0).astype(int) + 0.01 * np.less(x, 0).astype(int)
