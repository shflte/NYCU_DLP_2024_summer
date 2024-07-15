from fully_connected_layer import FullyConnectedLayer
import numpy as np
import matplotlib.pyplot as plt


class FullyConnectedNetwork:
    def __init__(self, epochs, input_unit=2, hidden_unit=4, output_unit=1,
                    learning_rate=0.1, activation='sigmoid', optimizer='sgd'):
        self.epochs = epochs
        self.input_unit = input_unit
        self.hidden_unit = hidden_unit
        self.output_unit = output_unit
        self.learning_rate = learning_rate
        self.activation = activation
        self.optimizer = optimizer
        self.layer1 = FullyConnectedLayer(input_unit, hidden_unit, activation, optimizer, learning_rate)
        self.layer2 = FullyConnectedLayer(hidden_unit, hidden_unit, activation, optimizer, learning_rate)
        self.layer3 = FullyConnectedLayer(hidden_unit, output_unit, activation, optimizer, learning_rate)
        self.loss_arr = []
        self.epoch_arr = []

    def forward(self, inputs):
        """
        Forward propagation:
            Since this network is a two-hidden-layer network, we need to compute 3 forward propagations.
            Forward the data from input layer to output layer.
        """
        hidden1 = self.layer1.forward(inputs)
        hidden2 = self.layer2.forward(hidden1)
        outputs = self.layer3.forward(hidden2)
        return outputs

    def backward(self, output_loss):
        """
        Backward propagation:
            Compute the gradient from output layer to input layer.
        """
        hidden1 = self.layer3.backward(output_loss)
        hidden2 = self.layer2.backward(hidden1)
        loss = self.layer1.backward(hidden2)
        return loss

    def optim(self):
        """
        Weight Update:
            Update the weight matrix for each layer.
        """
        self.layer1.optimization()
        self.layer2.optimization()
        self.layer3.optimization()

    def predict(self, inputs):
        """
        Predict:
            Predict the output based on the input data. 
            Since the output may not be 0 or 1, we have to round it to 0 or 1 to match the label. 
        """
        return np.round(self.forward(inputs))

    def MSELoss(self, predict_labels, labels):
        """
        MSELoss:
            Compute the MSE Loss based on the predicted result and the ground truth.
        """
        return np.mean((predict_labels - labels) ** 2)

    def derivativeMSELoss(self, predict_labels, labels):
        """
        Derivative MSELoss:
            Compute the derivative of MSE loss for back propagation.
        """
        return 2 * (predict_labels - labels) / len(labels)

    def train(self, inputs, labels):
        """
        Training:
            Train the model with 100000 epochs.
            1. Forward the input data to get the predicted resault
            2. Compute the MSE Loss between the predicted result and the ground truth.
            3. Back propagation
            4. Update the weight matrix

            Also, we have to show the loss every 5000 epochs.
        """
        for epoch in range(self.epochs):
            predicted_labels = self.forward(inputs)
            loss = self.MSELoss(predicted_labels, labels)
            self.backward(self.derivativeMSELoss(predicted_labels, labels))
            self.optim()
            if epoch % 100 == 0:
                self.loss_arr.append(loss)
                self.epoch_arr.append(epoch)
            if epoch % 5000 == 0:
                print(f'Epoch: {epoch}, Loss: {loss}')
    
    def test(self, inputs, labels):
        """
        Testing:
            Show the predicted result and the accuracy.
        """
        predicted_labels = self.forward(inputs)
        print(predicted_labels)
        accuracy, count = 0, 0
        for i in range(len(labels)):
            if np.round(predicted_labels[i]) == labels[i]:
                count += 1
        accuracy = float(count) / len(labels)
        print(f'Accuracy : {accuracy}')
    
    def show_result(self, inputs, labels):
        """
        Show result:
            Show the difference between predicted result and the ground truth.
        """
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.title('Ground Truth', fontsize=18)
        for i in range(inputs.shape[0]):
            if labels[i] == 0:
                plt.plot(inputs[i][0], inputs[i][1], 'ro')
            else:
                plt.plot(inputs[i][0], inputs[i][1], 'bo')
        
        pred_labels = self.predict(inputs)
        plt.subplot(1, 2, 2)
        plt.title('Predict Result', fontsize=18)
        for i in range(inputs.shape[0]):
            if pred_labels[i] == 0: 
                plt.plot(inputs[i][0], inputs[i][1], 'ro')
            else: 
                plt.plot(inputs[i][0], inputs[i][1], 'bo')
        plt.show()

    def show_learning_curve(self):
        """
        Show learning curve:
            Show the learning curve recorded while training.
        """
        plt.title('Learning Curve', fontsize=18)
        plt.plot(self.epoch_arr, self.loss_arr)
        plt.show()
