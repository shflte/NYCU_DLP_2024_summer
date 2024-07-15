from model import FullyConnectedNetwork
from argparse import ArgumentParser
import dataset


if __name__ =='__main__':
    """
        Parse the arguments from the command line
        Setup the model for training and testing
        Show the result and learning curve
    """
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='linear', help='linear/XOR dataset')
    parser.add_argument('-n', '--network', type=str, default='ffn', help='feed forward network/convolutional network')
    parser.add_argument('-e', '--epochs', type=int, default=100000, help='number of epochs')
    parser.add_argument('-u', '--units', type=int, default=4, help='number of hidden layer neurons')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.1, help='learning rate')
    parser.add_argument('-a', '--activation', type=str, default='sigmoid', help='sigmoid/tanh/relu/leaky relu')
    parser.add_argument('-o', '--optimizer', type=str, default='sgd', help='sgd/momentum/adagrad/adam')
    args = parser.parse_args()
    if args.dataset == 'linear':
        inputs, labels = dataset.generate_linear(n=100)
    elif args.dataset == 'xor':
        inputs, labels = dataset.generate_XOR_easy()
    epochs = args.epochs
    hidden_units = args.units
    learning_rate = args.learning_rate
    activation = args.activation
    optimizer = args.optimizer
    model = FullyConnectedNetwork(
                epochs=epochs,
                hidden_unit=hidden_units,
                learning_rate=learning_rate,
                activation=activation,
                optimizer=optimizer
            )

    model.train(inputs, labels)
    model.test(inputs, labels)
    model.show_result(inputs, labels)
    model.show_learning_curve()
