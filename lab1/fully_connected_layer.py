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
        Forward propagation: 
            output = sigma(W * x + b) 
        """
        self.input = inputs
        self.bias = 0.1 * np.ones(np.dot(self.input, self.weights).shape)
        if self.activation == 'none':
            self.output = np.dot(self.input, self.weights) + self.bias
        else:
            self.output = self.getActivation(np.dot(self.input, self.weights) + self.bias)
        return self.output
    
    def backward(self, loss):
        """
        Back propagation: 
            take the loss from the next layer and compute the loss of this layer
        """
        if self.activation == 'none':
            self.delta = loss
        else:
            self.delta = np.multiply(self.getDerivativeActivation(self.output), loss)
        return np.dot(self.delta, self.weights.T)
        
    def optimization(self):
        """
        Optimization: 
            Update the weight matrix based on the chosen optimizer
        """
        gradient = np.dot(self.input.T, self.delta)
        if self.optimizer == 'sgd':
            self.weights -= self.learning_rate * gradient
        elif self.optimizer == 'momentum':
            self.momentum = self.beta * self.momentum - self.learning_rate * gradient
            self.weights += self.momentum 
        elif self.optimizer == 'adagrad':
            gradient_square = np.sum(gradient ** 2)
            self.weights -= self.learning_rate * (1 / np.sqrt(gradient_square + self.epsilon)) * gradient 
        elif self.optimizer == 'adam':
            self.m_average = self.m_beta * self.m_average + (1 - self.m_beta) * gradient
            self.v_average = self.v_beta * self.v_average + (1 - self.v_beta) * (gradient ** 2)
            m_bias = self.m_average / (1 - self.m_beta)
            v_bias = self.v_average / (1 - self.v_beta)
            self.weights -= self.learning_rate * m_bias / np.sqrt(v_bias) + self.epsilon

    def getActivation(self, x):
        """
        Activation function:
            Given input x, output sigma(x), where sigma is the chosen activation function
        """
        if self.activation == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-x))
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return np.maximum(0.0, x)
        elif self.activation == 'leaky_relu':
            return np.maximum(0.0, x) + 0.01 * np.minimum(0.0, x)

    def getDerivativeActivation(self, x):
        """
        Derivative of the activation function:
            Given the output, compute the derivative of the chosen activation function
        """
        if self.activation == 'sigmoid':
            return np.multiply(x, 1.0 - x)
        elif self.activation == 'tanh':
            return 1 - x ** 2
        elif self.activation == 'relu':
            return np.greater(x, 0).astype(int)
        elif self.activation == 'leaky_relu':
            return np.greater(x, 0).astype(int) + 0.01 * np.less(x, 0).astype(int)
