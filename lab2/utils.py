# script for drawing figures, and more if needed
import matplotlib.pyplot as plt
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=300, help='the number of epochs')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('-o', '--optimizer', type=str, default='adam', help='optimizer')
    parser.add_argument('-b', '--batch-size', type=int, default=64, help='batch size')

    args = parser.parse_args()
    epochs = args.epochs
    learning_rate = args.learning_rate
    optimizer = args.optimizer
    batch_size = args.batch_size

    return epochs, learning_rate, optimizer, batch_size


def show_result(epoch_data, train_acc, test_acc, model):
    legends = []
    plt.title(str('Accuracy for ' + model))
    plt.xlabel('Epoches')
    plt.ylabel('Accuracy')

    plt.plot(epoch_data, train_acc)
    legends.append('train')
    plt.plot(epoch_data, test_acc)
    legends.append('test')

    plt.legend(legends)
    plt.show()


def show_learning_curve(epoch_data, train_loss, test_loss, model):
    legends = []
    plt.title(str('Learning Curve for ' + model))
    plt.xlabel('Epoches')
    plt.ylabel('Loss')

    plt.plot(epoch_data, train_loss)
    legends.append('train')
    plt.plot(epoch_data, test_loss)
    legends.append('test')

    plt.legend(legends)
    plt.show()
